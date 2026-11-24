"""J15: barrier batching, per-question equivalence, isolation and checkpoint flush."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.adapters import prompt_count
from drag_plus.flags import FlagError, Flags
from drag_plus.pipeline import DragPlusPipeline
from drag_plus.runner import fake
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.run import RunRequest, run_variant
from drag_plus.runner.variant import parse_variant
from drag_plus.scheduling.barrier import GENERATE, BarrierScheduler, Request
from fakes import FakeRetriever, IdentityTemplate, RecordingItem


def flags(size=1, **changes):
    values = {"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
              "question_batch_size": size, **changes}
    return Flags(values)


class CountingGenerator:
    """Deterministic content-driven generator that records each prompt batch."""

    def __init__(self):
        self.inner = fake.FakeGenerator()
        self.calls = []

    def generate(self, prompts, **params):
        self.calls.append(copy.deepcopy(prompts))
        batch = [prompts] if prompt_count(prompts) == 1 else list(prompts)
        return [fake.FakeGenerator._reply(self._text(prompt)) for prompt in batch]

    @staticmethod
    def _text(prompt):
        if isinstance(prompt, str):
            return prompt
        return "\n".join(message["content"] for message in prompt)


def run_questions(size, questions, **flag_changes):
    gen = CountingGenerator()
    pipeline = DragPlusPipeline(flags(size, **flag_changes), gen, FakeRetriever(),
                                IdentityTemplate(), "NQ")
    items = [RecordingItem(question, f"q{index}") for index, question in enumerate(questions)]
    return pipeline.run(items), items, gen


class FlagTests(unittest.TestCase):
    def test_batch_size_one_is_the_default_and_zero_is_rejected(self):
        self.assertEqual(Flags().question_batch_size, 1)
        with self.assertRaisesRegex(FlagError, ">= 1"):
            Flags({"question_batch_size": 0})


class EquivalenceTests(unittest.TestCase):
    QUESTIONS = tuple(f"question number {index}?" for index in range(9))

    def test_b1_transcripts_match_the_default_path(self):
        default, default_items, _ = run_questions(1, self.QUESTIONS[:2])
        explicit, explicit_items, _ = run_questions(1, self.QUESTIONS[:2])
        self.assertEqual([state.outputs for state in explicit],
                         [state.outputs for state in default])
        self.assertEqual([item.calls for item in explicit_items],
                         [item.calls for item in default_items])

    def test_batched_transcripts_equal_one_at_a_time(self):
        expected, expected_items, _ = run_questions(1, self.QUESTIONS)
        for size in (2, 4, 7):
            with self.subTest(size=size):
                actual, actual_items, _ = run_questions(size, self.QUESTIONS)
                self.assertEqual([state.outputs for state in actual],
                                 [state.outputs for state in expected])
                self.assertEqual([item.calls for item in actual_items],
                                 [item.calls for item in expected_items])

class AgentBatchTests(unittest.TestCase):
    def test_two_prompt_agent_requests_group_across_questions(self):
        questions = ("same question?",) * 2
        single, _, _ = run_questions(1, questions[:1], batch_agents_in_round=True)
        batched, _, gen = run_questions(2, questions, batch_agents_in_round=True)
        self.assertEqual([state.outputs for state in batched], [single[0].outputs] * 2)
        self.assertTrue(any(prompt_count(call) == 4 for call in gen.calls))

class CallCountTests(unittest.TestCase):
    def test_generate_calls_equal_barriers_not_questions_times_calls(self):
        questions = ("same question?",) * 3
        single, _, single_gen = run_questions(1, questions[:1])
        batched, _, batched_gen = run_questions(3, questions)
        self.assertEqual(len(batched_gen.calls), len(single_gen.calls))
        self.assertTrue(all(prompt_count(call) == 3 for call in batched_gen.calls))
        self.assertEqual([state.outputs for state in batched], [single[0].outputs] * 3)


class SchedulerTests(unittest.TestCase):
    def test_one_generate_per_barrier_and_finished_questions_leave(self):
        groups = []

        def execute(requests):
            groups.append(list(requests))
            for request in requests:
                request.result = ["ok"] * len(request.prompts)

        scheduler = BarrierScheduler(execute)

        def worker(rounds):
            def target():
                for _ in range(rounds):
                    scheduler.submit(Request(GENERATE, "query_agent", 0, "Proponent",
                                             prompts=["prompt"]))
            return target

        scheduler.run([(index, worker(1 if index == 0 else 3)) for index in range(3)])
        self.assertEqual(scheduler.stats.barriers, 3)
        self.assertEqual(scheduler.stats.generate_calls, 3)
        self.assertEqual([len(group) for group in groups], [3, 2, 2])


class BoomRetriever(FakeRetriever):
    def search(self, query):
        if query == "boom?":
            raise RuntimeError("index down")
        return super().search(query)


class ErrorIsolationTests(unittest.TestCase):
    def test_one_failed_question_does_not_stop_the_others(self):
        pipeline = DragPlusPipeline(flags(2), CountingGenerator(), BoomRetriever(),
                                    IdentityTemplate(), "NQ")
        items = [RecordingItem("boom?"), RecordingItem("fine?")]
        states = pipeline.run(items)
        self.assertEqual([state.status for state in states], ["failed", "done"])
        self.assertIn("index down", states[0].error)
        self.assertEqual(states[0].pred, "")
        self.assertIsNotNone(items[1].pred)

    def test_run_item_still_raises_the_single_question_failure(self):
        pipeline = DragPlusPipeline(flags(1), CountingGenerator(), BoomRetriever(),
                                    IdentityTemplate(), "NQ")
        with self.assertRaisesRegex(RuntimeError, "index down"):
            pipeline.run_item(RecordingItem("boom?"))


class SharedReferenceTests(unittest.TestCase):
    @staticmethod
    def mutable_ids(state):
        """Every list/dict/set object reachable from one state."""
        seen, found = set(), set()

        def walk(value):
            """Collect mutable container identities without revisiting a value."""
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, (list, dict, set)):
                found.add(id(value))
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(key)
                    walk(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item)

        walk(vars(state))
        return found

    def test_no_two_states_share_a_list_or_dict_object(self):
        states, _, _ = run_questions(2, ("same?", "same?"))
        self.assertIsNot(states[0], states[1])
        self.assertFalse(self.mutable_ids(states[0]) & self.mutable_ids(states[1]))

class StopRun(BaseException):
    """Simulates an operator interrupt after the first question was checkpointed."""


class RaggedGenerator:
    """q0 ends the query stage at round 0; q1 keeps debating; stop after q0's line."""

    def __init__(self, checkpoints: Path):
        self.checkpoints = checkpoints

    def generate(self, prompts, **params):
        if self.checkpoints.exists() and self.checkpoints.read_text().strip():
            raise StopRun()
        batch = [prompts] if prompt_count(prompts) == 1 else list(prompts)
        return [self.reply("\n".join(item) if isinstance(item, list) else item)
                for item in batch]

    @staticmethod
    def reply(text):
        if "You are the judge in a debate" in text:
            return "Proponent" if "q0?" in text else "unclear"
        if "critical thinker and debater" in text:
            return "argument" if "q0?" in text else "weak. Query Expansion: q1 more"
        return "The answer is: q0"


class CheckpointFlushTests(unittest.TestCase):
    ROWS = [{"dataset": "NQ", "split": "dev", "id": "q0", "question": "q0?",
             "golden_answers": ["q0"]},
            {"dataset": "NQ", "split": "dev", "id": "q1", "question": "q1?",
             "golden_answers": ["q1"]}]

    def test_each_finished_question_is_checkpointed_before_the_next_dispatch(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        out = Path(folder.name) / "run"
        backend = build_fake_backend()
        backend.generator = RaggedGenerator(out / "checkpoints.jsonl")
        variant = parse_variant({"name": "t", "arms": ["drag"], "flags": {
            "question_batch_size": 2, "max_query_debate_rounds": 3,
            "max_answer_debate_rounds": 1}})
        request = RunRequest(rows=self.ROWS, variant=variant, backend=backend, out_dir=out,
                             clock=lambda: 0.0, now=lambda: "2026-01-01T00:00:00Z",
                             environment=lambda: {}, command=[])
        with self.assertRaises(StopRun):
            run_variant(request)
        lines = [json.loads(line) for line in
                 (out / "checkpoints.jsonl").read_text().splitlines()]
        self.assertEqual([line["id"] for line in lines], ["q0"])

class FailureStatusTests(unittest.TestCase):
    ROWS = [{"dataset": "NQ", "split": "dev", "id": "q0", "question": "boom?",
             "golden_answers": ["x"]},
            {"dataset": "NQ", "split": "dev", "id": "q1", "question": "fine?",
             "golden_answers": ["x"]}]

    def test_an_isolated_failure_marks_the_run_failed(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        out = Path(folder.name) / "run"
        backend = build_fake_backend()
        backend.retriever = BoomRetriever()
        variant = parse_variant({"name": "t", "arms": ["drag"],
                                 "flags": {"question_batch_size": 2}})
        request = RunRequest(rows=self.ROWS, variant=variant, backend=backend, out_dir=out,
                             clock=lambda: 0.0, now=lambda: "2026-01-01T00:00:00Z",
                             environment=lambda: {}, command=[])
        manifest = run_variant(request)
        self.assertEqual(manifest["exit_status"], "failed")
        rows = [json.loads(line) for line in
                (out / "checkpoints.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["q0", "q1"])
        self.assertEqual(rows[0]["pred"], "")
        self.assertNotEqual(rows[1]["pred"], "")
