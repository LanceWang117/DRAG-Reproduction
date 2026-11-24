"""J16: refill window, batched retrieval, resume, and the offline throughput model."""
import json
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.flags import FlagError, Flags
from drag_plus.pipeline import DragPlusPipeline
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.run import RunRequest, run_variant
from drag_plus.runner.variant import parse_variant
from drag_plus.scheduling.barrier import SEARCH, BarrierScheduler, Request
from drag_plus.scheduling.refill import RefillWindow
from fakes import FakeRetriever, IdentityTemplate, RecordingItem
from simulate_throughput import CostModel, simulate, skewed_profile
from test_barrier_scheduler import CountingGenerator, RaggedGenerator, StopRun


def flags(size=1, **changes):
    values = {"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
              "question_batch_size": size, "refill_window": True, **changes}
    return Flags(values)


def run_questions(size, questions, retriever=None, **changes):
    gen = CountingGenerator()
    retriever = retriever or FakeRetriever()
    pipeline = DragPlusPipeline(flags(size, **changes), gen, retriever,
                                IdentityTemplate(), "NQ")
    items = [RecordingItem(question, f"q{index}") for index, question in enumerate(questions)]
    return pipeline.run(items), items, gen, retriever


class FlagTests(unittest.TestCase):
    def test_defaults_keep_baseline_and_validate_retrieval_cap(self):
        default = Flags()
        self.assertFalse(default.refill_window)
        self.assertEqual(default.retrieval_batch_size, 64)
        with self.assertRaisesRegex(FlagError, ">= 1"):
            Flags({"retrieval_batch_size": 0})


class RefillEquivalenceTests(unittest.TestCase):
    QUESTIONS = tuple(f"refill question {index}?" for index in range(9))

    def test_window_transcripts_equal_one_at_a_time(self):
        expected, expected_items, _, _ = run_questions(1, self.QUESTIONS)
        for size in (2, 8, 16):
            with self.subTest(size=size):
                actual, actual_items, _, _ = run_questions(size, self.QUESTIONS)
                self.assertEqual([state.outputs for state in actual],
                                 [state.outputs for state in expected])
                self.assertEqual([item.calls for item in actual_items],
                                 [item.calls for item in expected_items])


class BatchRecordingRetriever(FakeRetriever):
    """Record the query list of every ``batch_search`` the proxy issues."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def batch_search(self, queries):
        self.batches.append(list(queries))
        return super().batch_search(queries)


class BatchedRetrievalTests(unittest.TestCase):
    def test_all_ready_queries_share_one_batch_search(self):
        questions = ("first?", "second?", "third?")
        retriever = BatchRecordingRetriever()
        states, _, _, _ = run_questions(3, questions, retriever=retriever)
        self.assertEqual(retriever.batches[0], list(questions))
        self.assertTrue(all(state.status == "done" for state in states))

    def test_query_cap_splits_the_gathered_batch_and_stitches_results(self):
        questions = ("alpha?", "beta?", "gamma?")
        retriever = BatchRecordingRetriever()
        states, _, _, _ = run_questions(3, questions, retriever=retriever,
                                        retrieval_batch_size=2)
        self.assertEqual([len(batch) for batch in retriever.batches], [2, 1])
        self.assertEqual(retriever.batches[0] + retriever.batches[1], list(questions))
        self.assertTrue(all(state.status == "done" for state in states))


class SearchGatheringTests(unittest.TestCase):
    def test_requests_from_different_rounds_still_share_one_search_call(self):
        groups = []

        def execute(requests):
            groups.append(list(requests))
            for request in requests:
                request.result = [["doc"] for _ in request.queries]

        scheduler = BarrierScheduler(execute)

        def worker(round_index):
            def target():
                scheduler.submit(Request(SEARCH, "query", round_index,
                                         f"role{round_index}", queries=[f"q{round_index}"]))
            return target

        scheduler.run([(index, worker(index)) for index in range(3)])
        self.assertEqual(scheduler.stats.barriers, 1)
        self.assertEqual(scheduler.stats.search_calls, 1)
        self.assertEqual([len(group) for group in groups], [3])


class RefillWindowTests(unittest.TestCase):
    def test_admits_in_order_and_reports_occupancy(self):
        window = RefillWindow([("a", None), ("b", None), ("c", None)], 2)
        self.assertEqual([state for state, _ in window.fill(0)], ["a", "b"])
        self.assertFalse(window.exhausted)
        self.assertEqual([state for state, _ in window.fill(1)], ["c"])
        self.assertTrue(window.exhausted)
        self.assertEqual(window.admissions, 3)
        window.record(2)
        window.record(1)
        self.assertAlmostEqual(window.mean_occupancy, 0.75)

    def test_scheduler_refills_a_freed_slot_before_the_next_step(self):
        dispatched = []

        def execute(requests):
            dispatched.append(len(requests))
            for request in requests:
                request.result = ["ok"] * len(request.prompts)

        scheduler = BarrierScheduler(execute)
        lengths = {0: 1, 1: 1, 2: 4, 3: 4}

        def worker(length):
            def target():
                for _ in range(length):
                    scheduler.submit(Request("generate", "query", 0, "r", prompts=["p"]))
            return target

        pairs = [(index, worker(length)) for index, length in lengths.items()]
        window = RefillWindow(pairs, 2)
        scheduler.run([], refill=window)
        self.assertEqual(window.admissions, 4)
        self.assertEqual(max(dispatched), 2)
        self.assertGreaterEqual(window.mean_occupancy, 0.75)


class RefillResumeTests(unittest.TestCase):
    ROWS = [{"dataset": "NQ", "split": "dev", "id": f"q{index}",
             "question": f"q{index}?", "golden_answers": [f"q{index}"]}
            for index in range(3)]

    def request(self, out, backend, resume=False):
        variant = parse_variant({"name": "t", "arms": ["drag"], "flags": {
            "question_batch_size": 2, "refill_window": True,
            "max_query_debate_rounds": 3, "max_answer_debate_rounds": 1}})
        return RunRequest(rows=self.ROWS, variant=variant, backend=backend, out_dir=out,
                          resume=resume, clock=lambda: 0.0,
                          now=lambda: "2026-01-01T00:00:00Z",
                          environment=lambda: {}, command=[])

    def test_finished_question_is_skipped_and_unfinished_ones_rerun(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        out = Path(folder.name) / "run"
        first = build_fake_backend()
        first.generator = RaggedGenerator(out / "checkpoints.jsonl")
        with self.assertRaises(StopRun):
            run_variant(self.request(out, first))
        saved = [json.loads(line)["id"] for line in
                 (out / "checkpoints.jsonl").read_text().splitlines()]
        self.assertEqual(saved, ["q0"])

        gen = CountingGenerator()
        second = build_fake_backend()
        second.generator = gen
        manifest = run_variant(self.request(out, second, resume=True))
        self.assertEqual(manifest["exit_status"], "complete")
        rows = [json.loads(line) for line in
                (out / "checkpoints.jsonl").read_text().splitlines()]
        self.assertEqual({row["order"]: row["id"] for row in rows},
                         {0: "q0", 1: "q1", 2: "q2"})
        texts = ["\n".join(prompt) if isinstance(prompt, list) else prompt
                 for call in gen.calls for prompt in call]
        self.assertFalse(any("q0?" in text for text in texts))


class ThroughputModelTests(unittest.TestCase):
    def test_refill_keeps_the_window_full_until_the_queue_empties(self):
        profile = skewed_profile(64, seed=7)
        lockstep, refill = simulate(profile, 8, CostModel())
        self.assertGreaterEqual(refill.mean_occupancy, 0.90)
        self.assertLess(lockstep.mean_occupancy, refill.mean_occupancy)
        self.assertGreater(refill.questions_per_hour, lockstep.questions_per_hour)
        self.assertGreater(refill.max_occupancy, 0.99)

    def test_profile_is_deterministic_and_skewed(self):
        self.assertEqual(skewed_profile(64), skewed_profile(64))
        profile = skewed_profile(64)
        self.assertEqual(min(profile), 1)
        self.assertGreater(max(profile), 3)


if __name__ == "__main__":
    unittest.main()
