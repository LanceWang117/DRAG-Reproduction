"""The Naive arm, the answer policies and the variant file."""
import json
from pathlib import Path
import tempfile
import unittest

import frozen
from drag_plus.answers import native_pred, parse_naive, unified_pred
from drag_plus.flags import FlagError, Flags
from drag_plus.instrument.timing import CallRecorder
from drag_plus.naive import NaivePipeline
from drag_plus.runner.variant import VariantError, load_variant, parse_variant
from fakes import FakeRetriever, RecordingItem, ScriptedGenerator
from test_instrument import StepClock

PROJECT = Path(__file__).resolve().parents[2]


class RagTemplate:
    def __init__(self):
        self.calls = []

    def get_string(self, **kwargs):
        self.calls.append(kwargs)
        return f"PROMPT {kwargs['question']} | {len(kwargs['retrieval_result'])} docs"


def build(output, flags=None, dataset="NQ"):
    gen, ret, template = ScriptedGenerator([output]), FakeRetriever(), RagTemplate()
    flags = flags or Flags()
    pipeline = NaivePipeline(flags, gen, ret, template, dataset, CallRecorder(StepClock()))
    return pipeline, gen, ret, template


class NaivePipelineTests(unittest.TestCase):
    RAW = "The document says so. The answer is: Paris."

    def test_calls_match_sequential_pipeline_for_one_question(self):
        pipeline, gen, ret, template = build(self.RAW)
        item = RecordingItem("capital of France?")
        pipeline.run_item(item)
        self.assertEqual(ret.calls, ["capital of France?"])
        self.assertEqual(template.calls[0]["question"], "capital of France?")
        self.assertEqual(gen.calls, [("PROMPT capital of France? | 1 docs", {})])
        self.assertEqual([k for k, _ in item.calls],
                         ["retrieval_result", "prompt", "pred", "raw_pred", "pred"])

    def test_native_applies_the_baseline_parse_and_keeps_the_raw_text(self):
        pipeline = build(self.RAW)[0]
        item = RecordingItem("q?")
        state = pipeline.run_item(item)
        self.assertEqual((item.pred, item.output["raw_pred"]), ("Paris.", self.RAW))
        self.assertEqual((state.pred, state.raw_pred, state.status), ("Paris.", self.RAW, "done"))

    def test_unified_applies_the_shared_extractor(self):
        pipeline = build(self.RAW, Flags({"answer_extractor": "unified"}))[0]
        item = RecordingItem("q?")
        pipeline.run_item(item)
        self.assertEqual(item.pred, "Paris")

    def test_profiling_records_one_retrieval_and_one_generation_row(self):
        pipeline = build(self.RAW, Flags({"profile_calls": True}))[0]
        state = pipeline.run_item(RecordingItem("q?"))
        self.assertEqual([(r["stage"], r["batch_size"]) for r in state.calls],
                         [("retrieval", 1), ("answer_agent", 1)])

    def test_a_retriever_without_batch_search_is_rejected(self):
        class SearchOnly:
            def search(self, query):
                return []
        with self.assertRaisesRegex(TypeError, "batch_search"):
            NaivePipeline(Flags(), ScriptedGenerator([]), SearchOnly(), RagTemplate(), "NQ")


class AnswerPolicyTests(unittest.TestCase):
    CASES = ["The answer is: Paris", "no marker here", "a The answer is: X The answer is: Y",
             "The answer is:", "", "The answer is:  spaced  \n"]

    def test_parse_matches_the_frozen_single_agent_pred_parse(self):
        from model.utils import single_agent_pred_parse
        for text in self.CASES:
            with self.subTest(text=text):
                item = RecordingItem("q")
                item.output["pred"] = text
                single_agent_pred_parse([item])
                self.assertEqual(parse_naive(text), item.output["pred"])
                self.assertEqual(item.output["raw_pred"], text)

    def test_arms_differ_only_under_native(self):
        self.assertEqual(native_pred("drag", "The answer is: X"), "The answer is: X")
        self.assertEqual(native_pred("naive", "The answer is: X"), "X")
        self.assertEqual(unified_pred("The answer is: X.", "NQ"), "X")
        self.assertEqual(unified_pred("Yes, it is.", "StrategyQA"), "Yes")
        self.assertEqual(unified_pred("Yes, it is.", "NQ"), "Yes, it is")


class VariantTests(unittest.TestCase):
    def test_the_shipped_baseline_variant_uses_default_flags_and_both_arms(self):
        variant = load_variant(PROJECT / "improvements/configs/baseline.json")
        self.assertEqual(variant.flags, Flags())
        self.assertEqual((variant.name, variant.arms), ("baseline", ("drag", "naive")))

    def test_flags_and_arms_are_applied(self):
        variant = parse_variant({"name": "v1", "flags": {"max_answer_debate_rounds": 1},
                                 "arms": ["drag"], "seed": 3})
        self.assertEqual(variant.flags.max_answer_debate_rounds, 1)
        self.assertEqual((variant.arms, variant.seed), (("drag",), 3))

    def test_bad_variants_raise_clear_errors(self):
        bad = [(["x"], "JSON object"), ({"name": "a b"}, "name"), ({}, "name"),
               ({"name": "v", "armz": []}, "unknown variant key"),
               ({"name": "v", "arms": []}, "non-empty"),
               ({"name": "v", "arms": ["drag", "drag"]}, "distinct"),
               ({"name": "v", "arms": ["mad"]}, "distinct"),
               ({"name": "v", "flags": []}, "flags"), ({"name": "v", "seed": "1"}, "seed"),
               ({"name": "v", "seed": True}, "seed")]
        for data, message in bad:
            with self.subTest(data=data), self.assertRaisesRegex(VariantError, message):
                parse_variant(data)

    def test_an_unknown_flag_in_a_variant_is_a_flag_error(self):
        with self.assertRaises(FlagError):
            parse_variant({"name": "v", "flags": {"bogus": 1}})

    def test_invalid_json_file_is_a_variant_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v.json"
            path.write_text("{not json")
            with self.assertRaisesRegex(VariantError, "not valid JSON"):
                load_variant(path)
            path.write_text(json.dumps({"name": "ok"}))
            self.assertEqual(load_variant(path).name, "ok")


if __name__ == "__main__":
    unittest.main()
