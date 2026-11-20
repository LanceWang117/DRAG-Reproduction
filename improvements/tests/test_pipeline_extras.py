"""Behaviour T0 does not cover: the extractor flag, error path, isolation, parity."""
import contextlib
import io
import unittest

import frozen
from drag_plus import prompts
from drag_plus.flags import Flags
from drag_plus.judge_decision import resolve_judge_decision
from drag_plus.pipeline import DragPlusPipeline
from drag_plus.query_stage import QueryPoolError
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from t0_scenarios import answer_rounds, query_round


def build(outputs, flags=None, dataset="NQ", retriever=None):
    gen = ScriptedGenerator(outputs)
    pipeline = DragPlusPipeline(flags or Flags(), gen, retriever or FakeRetriever(),
                                IdentityTemplate(), dataset)
    return pipeline, gen


class ExtractorFlagTests(unittest.TestCase):
    OUTPUTS = query_round("Proponent") + answer_rounds(3)

    def run_with(self, extractor, outputs=None, dataset="NQ", rounds=3):
        flags = Flags({"answer_extractor": extractor, "max_answer_debate_rounds": rounds})
        pipeline, gen = build(outputs or self.OUTPUTS, flags, dataset)
        item = RecordingItem("who wrote it?")
        state = pipeline.run_item(item)
        return item, state, gen

    def test_native_keeps_the_raw_text_and_writes_no_raw_pred(self):
        item, state, _ = self.run_with("native")
        self.assertEqual(item.pred, "The answer is: moderator 2")
        self.assertNotIn("raw_pred", item.output)
        self.assertEqual(state.pred, state.raw_pred)

    def test_unified_extracts_pred_and_keeps_the_raw_text(self):
        item, state, _ = self.run_with("unified")
        self.assertEqual(item.pred, "moderator 2")
        self.assertEqual(item.output["raw_pred"], "The answer is: moderator 2")
        self.assertEqual(state.pred, "moderator 2")

    def test_the_extractor_never_changes_the_model_calls(self):
        native = self.run_with("native")[2].calls
        self.assertEqual(self.run_with("unified")[2].calls, native)

    def test_unified_maps_yes_no_for_strategyqa_only(self):
        outputs = query_round("Proponent") + answer_rounds(1)
        outputs[-1] = "Yes, because of the documents."
        item = self.run_with("unified", outputs, "StrategyQA", 1)[0]
        self.assertEqual(item.pred, "Yes")
        outputs = query_round("Proponent") + answer_rounds(1)
        outputs[-1] = "Yes, because of the documents."
        self.assertEqual(self.run_with("unified", outputs, "NQ", 1)[0].pred,
                         "Yes, because of the documents")


class IsolationTests(unittest.TestCase):
    def test_pipeline_keeps_no_per_question_state(self):
        outputs = query_round("Opponent", "x. Query Expansion: more") + \
            query_round("Proponent") + answer_rounds(2)
        pipeline, _ = build(outputs + outputs, Flags({"max_answer_debate_rounds": 2}))
        before = dict(vars(pipeline))
        first, second = RecordingItem("who wrote it?"), RecordingItem("who wrote it?")
        pipeline.run_item(first)
        self.assertEqual(vars(pipeline), before)
        pipeline.run_item(second)
        self.assertEqual(first.calls, second.calls)

    def test_state_records_outputs_decisions_and_key(self):
        pipeline, _ = build(query_round("Proponent") + answer_rounds(1),
                            Flags({"max_answer_debate_rounds": 1}))
        item = RecordingItem("who wrote it?")
        state = pipeline.run_item(item, key=("NQ", "dev", "q0"))
        self.assertEqual(state.outputs, item.calls)
        self.assertEqual(state.key, ("NQ", "dev", "q0"))
        self.assertEqual(state.decisions, [{"stage": "query", "round": 0,
                                            "decision": "Proponent", "status": "valid_initial"}])
        self.assertEqual((state.stage, state.status), ("answer", "active"))


class ErrorPathTests(unittest.TestCase):
    class Failing(FakeRetriever):
        def search(self, query):
            if "boom" in query:
                raise RuntimeError("index down")
            return super().search(query)

    def test_pool_failure_raises_a_typed_error_without_printing(self):
        outputs = query_round("Opponent", "weak. Query Expansion: boom")
        pipeline, _ = build(outputs, retriever=self.Failing())
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), self.assertRaises(QueryPoolError) as caught:
            pipeline.run_item(RecordingItem("who wrote it?"))
        self.assertEqual(captured.getvalue(), "")
        self.assertIsInstance(caught.exception, ValueError)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("boom", str(caught.exception))

    def test_unknown_agent_name_is_rejected(self):
        with self.assertRaises(ValueError):
            prompts.query_agent_system("Neutral Agent 0")


class VerdictParityTests(unittest.TestCase):
    OUTPUTS = ["Proponent", " opponent agent 1. ", "Opponent!", "Proponent Agent 1:\nOpponent",
               "both", "", None, "Proponent Agent", "OPPONENT", "Winner: Proponent"]

    def test_copied_resolver_matches_the_frozen_module(self):
        for output in self.OUTPUTS:
            for retry in ("Opponent", "garbage"):
                with self.subTest(output=output, retry=retry):
                    ours = resolve_judge_decision(output, lambda: retry)
                    theirs = frozen.frozen_judge.resolve_judge_decision(output, lambda: retry)
                    self.assertEqual((ours.decision, ours.status, ours.retry_output),
                                     (theirs.decision, theirs.status, theirs.retry_output))


if __name__ == "__main__":
    unittest.main()
