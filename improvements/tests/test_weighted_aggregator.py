"""J19 weighted aggregator, weights loader and scorer hook, on scripted data."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "improvements"))

from drag_plus.flags import FlagError, Flags  # noqa: E402
from drag_plus.gates.features import FEATURE_NAMES  # noqa: E402
from drag_plus.grounding.scorer import (AggregatorWeights,  # noqa: E402
                                        FaithfulnessConfigError, agreement,
                                        load_weights, select_weighted)
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402

DIGEST = "a" * 64
PARIS = [{"id": "d1", "contents": "Paris is the capital of France."}]
LONDON = [{"id": "d2", "contents": "London is the capital of England."}]


def _write_weights(path, conf=0.0, s_faith=1.0, agr=0.0, split="train", digest=DIGEST):
    data = {"weights": {"conf": conf, "s_faith": s_faith, "agr": agr},
            "training_split": split, "manifest_sha256": digest}
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _run(outputs, docs=None, **flags):
    values = {"max_query_debate_rounds": 0, "max_answer_debate_rounds": 1}
    values.update(flags)
    generator = ScriptedGenerator(outputs)
    pipeline = DragPlusPipeline(Flags(values), generator,
                                FakeRetriever({} if docs is None else docs),
                                IdentityTemplate(), "NQ")
    item = RecordingItem("q")
    return pipeline.run_item(item), generator, item


def _signals(state):
    rows = dict(state.outputs)["AnswerStage_Aggregator_Signals"]
    return {row["answer"]: row for row in rows}


class SelectWeightedTests(unittest.TestCase):
    def setUp(self):
        self.weights = AggregatorWeights(
            conf=0.0, s_faith=1.0, agr=0.0,
            training_split="train", manifest_sha256=DIGEST)

    def test_faithfulness_overrules_a_plurality(self):
        answers = [("a", "Paris"), ("b", "Paris"), ("c", "London")]
        scorer = lambda answer, passages: 1.0 if answer == "London" else 0.0
        answer, reason, rows = select_weighted(answers, ["London"], scorer, self.weights)
        self.assertEqual((answer, reason), ("London", "weighted"))
        self.assertEqual({row["answer"]: row["s_faith"] for row in rows},
                         {"Paris": 0.0, "London": 1.0})

    def test_confidence_can_win_when_faithfulness_is_neutral(self):
        weights = AggregatorWeights(conf=1.0, s_faith=0.0, agr=0.0,
                                    training_split="train", manifest_sha256=DIGEST)
        answers = [("a", "Paris"), ("b", "Paris"), ("c", "London")]
        answer, reason, _ = select_weighted(answers, [], lambda a, p: 0.5, weights)
        self.assertEqual((answer, reason), ("Paris", "weighted"))

    def test_exact_tie_keeps_the_first_seen_answer(self):
        weights = AggregatorWeights(conf=0.0, s_faith=0.0, agr=1.0,
                                    training_split="train", manifest_sha256=DIGEST)
        answer, reason, _ = select_weighted(
            [("a", "Paris"), ("b", "London")], [], lambda a, p: 0.0, weights)
        self.assertEqual((answer, reason), ("Paris", "first_seen"))

    def test_empty_candidates_are_an_empty_result(self):
        self.assertEqual(select_weighted([], [], lambda a, p: 0.0, self.weights),
                         ("", "empty", []))

    def test_non_finite_scorer_is_a_configuration_error(self):
        with self.assertRaisesRegex(FaithfulnessConfigError, "non-finite"):
            select_weighted([("a", "Paris")], [], lambda a, p: math.nan, self.weights)

    def test_agreement_is_soft_token_overlap(self):
        self.assertEqual(agreement("Paris", ["London"]), 0.0)
        self.assertEqual(agreement("Paris", []), 1.0)
        self.assertAlmostEqual(agreement("New York", ["New York City"]), 0.8)


class WeightsLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def path(self, name="weights.json"):
        return Path(self.temp.name) / name

    def test_roundtrip_records_provenance(self):
        loaded = load_weights(_write_weights(self.path()))
        self.assertEqual((loaded.conf, loaded.s_faith, loaded.agr), (0.0, 1.0, 0.0))
        self.assertEqual((loaded.training_split, loaded.manifest_sha256),
                         ("train", DIGEST))

    def test_refuses_the_test_split(self):
        with self.assertRaisesRegex(FaithfulnessConfigError, "test split"):
            load_weights(_write_weights(self.path(), split="TEST"))

    def test_rejects_unknown_missing_or_bad_weights(self):
        for weights in ({"conf": 0.5, "s_faith": 0.5},
                        {"conf": 0.5, "s_faith": 0.5, "agr": 0.5, "extra": 0.0}):
            with self.subTest(weights=weights), self.assertRaisesRegex(
                    FaithfulnessConfigError, "exactly"):
                load_weights(self._custom(weights=weights))
        with self.assertRaisesRegex(FaithfulnessConfigError, "finite and >= 0"):
            load_weights(self._custom(weights={"conf": -0.1, "s_faith": 1.0, "agr": 0.0}))
        with self.assertRaisesRegex(FaithfulnessConfigError, "positive total"):
            load_weights(self._custom(weights={"conf": 0.0, "s_faith": 0.0, "agr": 0.0}))

    def test_rejects_a_malformed_manifest_hash(self):
        with self.assertRaisesRegex(FaithfulnessConfigError, "manifest_sha256"):
            load_weights(_write_weights(self.path(), digest="xyz"))

    def test_rejects_missing_or_non_object_files(self):
        with self.assertRaisesRegex(FaithfulnessConfigError, "cannot read"):
            load_weights(self.path("absent.json"))
        path = self.path("list.json")
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(FaithfulnessConfigError, "JSON object"):
            load_weights(path)

    def _custom(self, **overrides):
        data = {"weights": {"conf": 0.4, "s_faith": 0.4, "agr": 0.2},
                "training_split": "train", "manifest_sha256": DIGEST}
        data.update(overrides)
        path = self.path("custom.json")
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


class WeightedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_evidence_support_picks_the_grounded_answer(self):
        weights = _write_weights(Path(self.temp.name) / "weights.json")
        state, generator, _ = _run(
            ["The answer is: Paris", "The answer is: London"],
            docs={"q": LONDON}, aggregator="weighted", aggregator_weights=weights)
        self.assertEqual(state.pred, "London")
        result = dict(state.outputs)["AnswerStage_Aggregator_Result"]
        self.assertEqual((result["aggregator"], result["tie_break"]), ("weighted", "weighted"))
        signals = _signals(state)
        self.assertEqual(signals["London"]["s_faith"], 1.0)
        self.assertEqual(signals["Paris"]["s_faith"], 0.0)
        weights_log = dict(state.outputs)["AnswerStage_Aggregator_Weights"]
        self.assertEqual(weights_log["scorer"], "heuristic")
        self.assertEqual(weights_log["training_split"], "train")
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual([row["conf"] for row in signals.values()], [0.5, 0.5])

    def test_default_scorer_is_the_heuristic(self):
        weights = _write_weights(Path(self.temp.name) / "weights.json")
        state, _, _ = _run(["The answer is: Paris", "The answer is: London"],
                           docs={"q": PARIS}, aggregator="weighted",
                           aggregator_weights=weights)
        self.assertEqual(state.pred, "Paris")
        self.assertNotIn("minicheck", sys.modules)

    def test_unified_extraction_feeds_the_scorer(self):
        weights = _write_weights(Path(self.temp.name) / "weights.json")
        state, _, _ = _run(["The answer is: **London**.", "The answer is: London."],
                           docs={"q": LONDON}, aggregator="weighted",
                           aggregator_weights=weights)
        self.assertEqual(state.pred, "London")
        self.assertEqual(_signals(state)["London"]["conf"], 1.0)

    def test_weighted_needs_a_weights_file(self):
        with self.assertRaisesRegex(FlagError, "requires aggregator_weights"):
            Flags({"aggregator": "weighted"})

    def test_gate_features_use_the_configured_scorer(self):
        model = {"feature_names": list(FEATURE_NAMES),
                 "coefficients": [0.0, 0.0, 0.0, 0.0], "intercept": 0.0,
                 "calibration": {"kind": "identity"}, "training_split": "train",
                 "manifest_hash": "abc"}
        path = Path(self.temp.name) / "gate.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        state, _, _ = _run(
            ["The answer is: Paris France", "The answer is: Paris France",
             "The answer is: Paris"],
            docs={"q": [{"contents": "Paris is the capital of France."}]},
            response_rounds=1, moderator_final_only=True,
            escalation_gate="risk", escalation_model=str(path),
            escalation_tau=0.5, faithfulness_scorer="heuristic")
        self.assertEqual(state.gate_features["evidence_support"], 0.0)


if __name__ == "__main__":
    unittest.main()
