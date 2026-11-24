"""J14: fitting, cross-validating and calibrating the escalation gate from transcripts.

Synthetic data with a known signal must recover an AUROC above the stated threshold and
write the reliability figure. The test split is refused, a no-signal input fails the
threshold, and transcript features never read the gold label.
"""
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "improvements"))
sys.path.insert(0, str(ROOT / "scripts"))
import fit_gate  # noqa: E402
from drag_plus.gates.features import FEATURE_NAMES  # noqa: E402


def synthetic(count=300, seed=7):
    """A strong latent signal: two features drive P(wrong), the rest are neutral."""
    rng = random.Random(seed)
    records = []
    for _ in range(count):
        first, second = rng.random(), rng.random()
        logit = 5.0 * (first - 0.5) + 4.0 * (second - 0.5)
        probability = 1.0 / (1.0 + math.exp(-logit))
        records.append({"features": {"disagreement": first, "cluster_entropy": second,
                                     "s_bar_O": 0.5, "evidence_support": 0.5},
                        "wrong": rng.random() < probability})
    return records


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8")
    return path


def run_fit(directory, records, *extra):
    source = write_jsonl(Path(directory) / "records.jsonl", records)
    model, figure = Path(directory) / "model.json", Path(directory) / "reliability.svg"
    code = fit_gate.main(["--input", str(source), "--out", str(model),
                          "--figure", str(figure), *extra])
    return code, model, figure


class FitScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_synthetic_signal_recovers_a_useful_calibrated_model(self):
        code, model_path, figure_path = run_fit(self.temp.name, synthetic())
        self.assertEqual(code, 0)
        model = json.loads(model_path.read_text(encoding="utf-8"))
        self.assertEqual(model["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(model["training_split"], "train")
        self.assertEqual(model["calibration"]["kind"], "platt")
        self.assertGreaterEqual(model["metrics"]["auroc_oof"], 0.70)
        self.assertGreaterEqual(model["metrics"]["auroc_calibrated"], 0.65)
        self.assertEqual(len(model["manifest_hash"]), 64)
        figure = figure_path.read_text(encoding="utf-8")
        self.assertIn("<svg", figure)
        self.assertIn("polyline", figure)
        self.assertIn("Reliability", figure)

    def test_test_split_is_refused_before_any_output(self):
        code, model_path, _figure = run_fit(self.temp.name, synthetic(), "--split", "test")
        self.assertEqual(code, 2)
        self.assertFalse(model_path.exists())

    def test_no_signal_fails_the_stated_threshold(self):
        records = [{"features": {name: 0.0 for name in FEATURE_NAMES},
                    "correct": index % 2 == 0} for index in range(200)]
        code, model_path, _figure = run_fit(self.temp.name, records)
        self.assertEqual(code, 1)
        model = json.loads(model_path.read_text(encoding="utf-8"))
        self.assertLess(model["metrics"]["auroc_oof"], 0.6)

    def test_low_auroc_threshold_is_configurable(self):
        records = [{"features": {name: 0.0 for name in FEATURE_NAMES},
                    "correct": index % 2 == 0} for index in range(200)]
        code, _model, _figure = run_fit(self.temp.name, records, "--min-auroc", "0.4")
        self.assertEqual(code, 0)


class TranscriptFeatureTests(unittest.TestCase):
    def test_features_are_recomputed_from_the_transcript_fields(self):
        record = {"agent_answers": ["The answer is: Paris", "The answer is: London"],
                  "query_pool": {"q": [{"contents": "Paris is in France"}]},
                  "judge_score": 0.0}
        features = fit_gate.features_from_record(record)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features["disagreement"], 0.5)
        self.assertAlmostEqual(features["cluster_entropy"], 1.0)

    def test_gold_label_is_used_only_for_the_target(self):
        record = {"features": {name: 0.5 for name in FEATURE_NAMES}, "wrong": False}
        self.assertFalse(fit_gate._wrong(record))
        record = {"features": {name: 0.5 for name in FEATURE_NAMES},
                  "pred": "The answer is: Paris", "gold_answers": ["London"]}
        self.assertTrue(fit_gate._wrong(record))

    def test_a_feature_object_must_have_exactly_the_four_names(self):
        with self.assertRaises(ValueError):
            fit_gate.features_from_record({"features": {"disagreement": 0.0}})


class AurocTests(unittest.TestCase):
    def test_perfect_and_tied_rankings(self):
        self.assertEqual(fit_gate.auroc([0.0, 0.1, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(fit_gate.auroc([0.5, 0.5], [0, 1]), 0.5)
        self.assertEqual(fit_gate.auroc([0.0, 1.0], [0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
