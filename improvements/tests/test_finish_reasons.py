"""J24: observed finish reasons in timing rows, scorer wiring and the full stack offline."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.adapters import Backends, last_finish_reasons
from drag_plus.flags import Flags
from drag_plus.instrument.timing import STAGES, CallInfo, CallRecorder, observed_reason
from drag_plus.judge.logprob import ScoredGenerator, needs_scores
from drag_plus.pipeline import DragPlusPipeline
from drag_plus.runner import fake
from drag_plus.runner.run import run_variant
from drag_plus.runner.variant import parse_variant
from fakes import FakeRetriever, IdentityTemplate, RecordingItem
from test_instrument import StepClock
from test_runner_fake import make_request, read_lines

FULL_STACK = {
    "answer_extractor": "unified", "moderator_final_only": True,
    "batch_agents_in_round": True, "question_batch_size": 4, "refill_window": True,
    "search_via_batch": True,
    "skip_duplicate_query": True, "zero_novelty_exit": True, "response_rounds": 1,
    "agent_max_tokens": 128, "moderator_max_tokens": 32, "concise_agents": True,
    "evidence_store": "anchored", "evidence_delimiters": True,
    "rewrite_novelty_audit": True, "context_token_budget": 2500,
    "prompt_layout": "evidence_first", "judge_scoring": "logprob",
    "judge_order_balance": True, "opponent_action_schema": True,
    "judge_sees_refined_query": True, "lazy_query_opponent": True,
    "opponent_mode": "grounded", "verdict_sequestration": True, "aggregator": "vote",
    "escalation_tau": 0.0}


class ReportingGenerator(fake.FakeGenerator):
    """The fake generator plus a per-prompt reason: ``length`` for prompts about beta."""

    def __init__(self):
        self.last_finish_reasons = None
        self.batches = []

    def generate(self, prompts, **params):
        outputs = super().generate(prompts, **params)
        batch = prompts if len(outputs) > 1 else [prompts]
        self.batches.append(len(outputs))
        self.last_finish_reasons = ["length" if "beta" in str(p) else "stop" for p in batch]
        return outputs


class ObservedReasonTests(unittest.TestCase):
    def test_length_wins_agreement_is_kept_and_disagreement_is_mixed(self):
        self.assertEqual(observed_reason(["stop", "length"]), "length")
        self.assertEqual(observed_reason(["stop", "stop"]), "stop")
        self.assertEqual(observed_reason(["stop", "scored"]), "mixed")
        self.assertEqual(observed_reason(["stop", "unknown"]), "stop")
        self.assertIsNone(observed_reason(["unknown", "unknown"]))
        self.assertIsNone(observed_reason([]))
        self.assertIsNone(observed_reason(None))

    def test_the_observed_reason_beats_the_budget_inference(self):
        rows = []
        info = CallInfo("answer_agent", 0, "proponent", rows)
        recorder = CallRecorder(StepClock())
        recorder.generation(info, "p", lambda: "one two", limit=None,
                            reasons=lambda: ["length"])
        recorder.generation(info, "p", lambda: "one two", limit=2, reasons=lambda: None)
        recorder.generation(info, "p", lambda: "one two", limit=None)
        self.assertEqual([row["finish_reason"] for row in rows], ["length", "length", None])

    def test_reasons_are_used_only_when_the_generator_reports_one_per_prompt(self):
        generator = SimpleNamespace(last_finish_reasons=["stop", "length"])
        self.assertEqual(last_finish_reasons(generator, 2), ["stop", "length"])
        self.assertIsNone(last_finish_reasons(generator, 1))
        self.assertIsNone(last_finish_reasons(SimpleNamespace(), 1))
        self.assertIsNone(last_finish_reasons(SimpleNamespace(last_finish_reasons=[1]), 1))

    def test_backends_record_the_generator_reason_for_one_call_and_a_batch(self):
        rows = []
        backends = Backends(ReportingGenerator(), FakeRetriever(), IdentityTemplate(),
                            CallRecorder(StepClock()))
        info = CallInfo("answer_agent", 0, "proponent", rows)
        backends.complete([{"role": "user", "content": "beta"}], call=info)
        backends.complete_batch([[{"role": "user", "content": "alpha"}],
                                 [{"role": "user", "content": "beta"}]], call=info)
        backends.complete([{"role": "user", "content": "alpha"}], call=info)
        self.assertEqual([row["finish_reason"] for row in rows], ["length", "length", "stop"])

    def test_one_conversation_is_one_scored_prompt_in_its_row(self):
        class Scorer(fake.FakeGenerator):
            def generate_with_scores(self, prompts, labels, **params):
                self.last_finish_reasons = ["scored"]
                return super().generate_with_scores(prompts, labels, **params)

        rows = []
        backends = Backends(Scorer(), FakeRetriever(), IdentityTemplate(),
                            CallRecorder(StepClock()))
        messages = [{"role": "system", "content": "judge"}, {"role": "user", "content": "q"}]
        backends.complete_with_scores(messages, labels=("Proponent", "Opponent"),
                                      call=CallInfo("query_judge", 0, "judge", rows))
        self.assertEqual((rows[0]["batch_size"], rows[0]["finish_reason"]), (1, "scored"))


class GroupedBatchTests(unittest.TestCase):
    def test_each_question_gets_its_own_slice_of_a_grouped_call(self):
        generator = ReportingGenerator()
        flags = Flags({"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
                       "question_batch_size": 2, "search_via_batch": True,
                       "profile_calls": True})
        pipeline = DragPlusPipeline(flags, generator, FakeRetriever(), IdentityTemplate(),
                                    "NQ", recorder=CallRecorder(StepClock()))
        states = pipeline.run([RecordingItem("alpha question?", "q0"),
                               RecordingItem("beta question?", "q1")])
        self.assertIn(2, generator.batches)
        alpha, beta = ([row["finish_reason"] for row in state.calls
                        if row["stage"] != "retrieval"] for state in states)
        self.assertTrue(alpha and beta)
        self.assertEqual(set(alpha), {"stop"})
        self.assertEqual(set(beta), {"length"})


class ScorerWiringTests(unittest.TestCase):
    def test_the_vllm_wrapper_is_used_for_the_judge_or_the_lazy_opponent(self):
        self.assertFalse(needs_scores(Flags()))
        self.assertTrue(needs_scores(Flags({"judge_scoring": "logprob"})))
        self.assertTrue(needs_scores(Flags({"lazy_query_opponent": True,
                                            "escalation_tau": 0.5})))

    def test_the_vllm_scorer_reports_raw_reasons_and_clears_them_for_text(self):
        def raw(reason):
            completion = SimpleNamespace(text="Opponent", logprobs=[], finish_reason=reason)
            return SimpleNamespace(outputs=[completion])

        class Inner:
            tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [len(text)])

            def generate(self, prompts, **params):
                if params.get("return_raw_output"):
                    return [raw("length"), raw(None)]
                return ["text"]

        scorer = ScoredGenerator(Inner())
        scorer.generate_with_scores(["a", "b"], labels=("Proponent", "Opponent"))
        self.assertEqual(scorer.last_finish_reasons, ["scored", "scored"])
        scorer.generate("a")
        self.assertIsNone(scorer.last_finish_reasons)


class FullStackTests(unittest.TestCase):
    def run_stack(self, tau):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            variant = parse_variant({"name": "full-stack",
                                     "flags": dict(FULL_STACK, escalation_tau=tau)})
            manifest = run_variant(make_request(out, variant=variant))
            transcripts = read_lines(out / "transcripts.jsonl")
            timings = read_lines(out / "stage-timings.jsonl")
        self.assertEqual(manifest["exit_status"], "complete")
        drag = [row for row in transcripts if row["arm"] == "drag"]
        self.assertEqual(len(drag), 5)
        self.assertTrue(all(row["pred"] for row in drag))
        for row in timings:
            self.assertIn(row["stage"], STAGES)
        decisions = [entry["value"] for row in drag for entry in row["entries"]
                     if entry["key"].startswith("QueryStage_Sufficiency_Round0_Decision")]
        stages = {row["stage"] for row in timings if row["arm"] == "drag"}
        return decisions, stages

    def test_the_lazy_opponent_can_end_every_query_stage_offline(self):
        decisions, stages = self.run_stack(0.0)
        self.assertEqual(decisions, ["sufficient"] * 5)
        self.assertNotIn("query_judge", stages)

    def test_the_log_prob_judge_runs_without_a_retry_offline(self):
        decisions, stages = self.run_stack(1.0)
        self.assertEqual(decisions, ["insufficient"] * 5)
        self.assertTrue({"query_sufficiency", "query_agent", "query_judge"} <= stages)
        self.assertNotIn("query_judge_retry", stages)


if __name__ == "__main__":
    unittest.main()
