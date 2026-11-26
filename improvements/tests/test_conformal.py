"""J21: conformal calibration, prediction sets, selective metrics and pipeline step.

For 200 exchangeable synthetic seeds the test calibrates on 99 fresh points and checks
whether the next label is in the set; the exact Clopper-Pearson interval of the observed
coverage must contain ``1 - alpha``. Small-n, pooled fallback, loader refusals, metrics
and pipeline integration are separate. ``conformal_alpha`` defaults to null (T0).
"""
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

IMPROVEMENTS = Path(__file__).resolve().parents[1]
PROJECT = IMPROVEMENTS.parent
sys.path.insert(0, str(IMPROVEMENTS))
sys.path.insert(0, str(PROJECT / "scripts"))
from drag_plus.flags import FlagError, Flags  # noqa: E402
from drag_plus.gates.conformal import (CalibrationError, calibrate, cluster_key,  # noqa: E402
                                       conformal_quantile, load_calibration, prediction_set,
                                       probability_distribution, select_threshold,
                                       state_distribution)
from drag_plus.gates.conformal_metrics import clopper_pearson, summarise  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from drag_plus.state import QuestionState  # noqa: E402
import eval_conformal  # noqa: E402
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402

ALPHA = 0.1


def write_calibration(directory, qhat=1.0, dataset=None, **overrides):
    data = {"format_version": 1, "alpha": ALPHA, "split": "dev", "min_n": 20,
            "pooled": {"n": 10, "qhat": qhat}, "datasets": {}}
    if dataset is not None:
        data["datasets"][dataset] = {"n": 30, "qhat": qhat}
    data.update(overrides)
    path = Path(directory) / "conformal.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def distribution(values): return {str(index): value for index, value in enumerate(values)}


def draw(rng):
    """Three exchangeable class probabilities with no degenerate classes."""
    weights = [math.exp(rng.gauss(0.0, 1.0)) for _ in range(3)]
    total = sum(weights)
    return [weight / total for weight in weights]


def examples(dataset, count, score): return [(dataset, "gold", {"gold": 1.0 - score, "other": score})] * count


class QuantileTests(unittest.TestCase):
    def test_rank_uses_the_ceiling_correction(self):
        self.assertEqual(conformal_quantile([0.0, 0.5, 1.0], 0.5), 0.5)
        self.assertEqual(conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.5), 0.3)
        self.assertEqual(conformal_quantile([0.0, 0.4], 0.4), 0.4)

    def test_too_small_a_calibration_set_abstains_always(self):
        self.assertEqual(conformal_quantile([0.0] * 5, 0.1), 1.0)
        self.assertEqual(conformal_quantile([], 0.1), 1.0)
        self.assertEqual(prediction_set({"paris": 0.6, "london": 0.4}, 1.0),
                         ["london", "paris"])

    def test_alpha_and_probability_edges(self):
        for bad in (0, 1, -0.1, 1.1, True, "0.1"):
            with self.subTest(alpha=bad), self.assertRaises(ValueError):
                conformal_quantile([0.1], bad)


class DistributionTests(unittest.TestCase):
    def test_votes_support_and_judge_margin_change_p_hat(self):
        neutral = probability_distribution({"paris": 2, "london": 1}, {}, None)
        self.assertAlmostEqual(sum(neutral.values()), 1.0)
        self.assertGreater(neutral["paris"], neutral["london"])
        supported = probability_distribution({"paris": 2, "london": 1}, {"london": 1.0}, None)
        self.assertGreater(supported["london"], neutral["london"])
        decisive = probability_distribution({"paris": 2, "london": 1}, {}, 4.0)
        self.assertGreater(max(decisive.values()), max(neutral.values()))
        self.assertEqual(probability_distribution({}, {"paris": 1.0}, 0.0), {})

    def test_clusters_use_extraction_and_j01_normalisation(self):
        self.assertEqual(cluster_key("The answer is: Paris."), "paris")
        self.assertEqual(cluster_key("**Paris**"), "paris")
        self.assertEqual((cluster_key("Yes", True), cluster_key("")), ("yes", ""))

    def test_state_distribution_counts_agents_and_the_verdict(self):
        state = QuestionState("who wrote it?", pred="Paris",
                              answer_messages={"Proponent": [{"content": "The answer is: Paris"}],
                                               "Opponent": [{"content": "The answer is: London"}]})
        scores = state_distribution(state, "NQ")
        expected = probability_distribution({"paris": 2, "london": 1}, {}, None)["paris"]
        self.assertAlmostEqual(scores["paris"], expected)
        self.assertAlmostEqual(sum(scores.values()), 1.0)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_dataset_threshold_and_pooled_fallback(self):
        document = calibrate(examples("NQ", 30, 0.2) + examples("HotpotQA", 3, 0.9),
                             ALPHA, min_n=20)
        path = Path(self.temp.name) / "cal.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        calibration = load_calibration(path, ALPHA)
        self.assertEqual(select_threshold(calibration, "NQ"),
                         (document["datasets"]["NQ"]["qhat"], False))
        pooled, used_pooled = select_threshold(calibration, "HotpotQA")
        self.assertTrue(used_pooled)
        self.assertEqual(pooled, document["pooled"]["qhat"])

    def test_small_dataset_scores_one(self):
        document = calibrate(examples("NQ", 5, 0.0), ALPHA, min_n=20)
        self.assertEqual(document["datasets"]["NQ"]["qhat"], 1.0)

    def test_loader_refuses_test_split_alpha_mismatch_and_bad_numbers(self):
        cases = ({"split": "test"}, {"alpha": 0.2}, {"format_version": 2},
                 {"pooled": {"n": 1, "qhat": 2.0}},
                 {"datasets": {"NQ": {"n": -1, "qhat": 0.1}}})
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(CalibrationError):
                load_calibration(write_calibration(self.temp.name, **overrides), ALPHA)
        with self.assertRaises(CalibrationError):
            calibrate(examples("NQ", 3, 0.1), ALPHA, split="test")
        with self.assertRaises(ValueError):
            calibrate([], ALPHA)


class CoverageTests(unittest.TestCase):
    def test_empirical_coverage_matches_nominal_for_200_exchangeable_seeds(self):
        rng = random.Random(20261126)
        for alpha in (0.1, 0.2):
            with self.subTest(alpha=alpha):
                covered = 0
                for _ in range(200):
                    scores = []
                    for _ in range(99):
                        probabilities = draw(rng)
                        label = rng.choices(range(3), weights=probabilities)[0]
                        scores.append(1.0 - probabilities[label])
                    probabilities = draw(rng)
                    label = rng.choices(range(3), weights=probabilities)[0]
                    answer_set = prediction_set(distribution(probabilities),
                                                conformal_quantile(scores, alpha))
                    covered += str(label) in answer_set
                low, high = clopper_pearson(covered, 200)
                self.assertLessEqual(low, 1.0 - alpha, (covered, low, high))
                self.assertGreaterEqual(high, 1.0 - alpha, (covered, low, high))


class MetricsTests(unittest.TestCase):
    def test_clopper_pearson_edges(self):
        self.assertEqual(clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(clopper_pearson(10, 10)[1], 1.0)
        low, high = clopper_pearson(5, 10)
        self.assertAlmostEqual(low, 0.187, places=3)
        self.assertAlmostEqual(high, 0.813, places=3)
        with self.assertRaises(ValueError):
            clopper_pearson(11, 10)

    def test_abstentions_leave_full_coverage_accuracy_but_not_selective(self):
        rows = [
            {"dataset": "NQ", "covered": True, "set_size": 1, "abstained": False,
             "answered_correct": False, "argmax_correct": False, "confidence": 0.9, "fold": 0},
            {"dataset": "NQ", "covered": False, "set_size": 2, "abstained": True,
             "answered_correct": False, "argmax_correct": True, "confidence": 0.8, "fold": 0},
        ]
        report = summarise(rows, {(0, target): 0.0 for target in (0.5, 0.7, 0.9)})
        self.assertEqual(report["full_coverage_accuracy"], 0.5)
        self.assertEqual(report["selective_accuracy"], 0.0)
        self.assertEqual(report["abstention_rate"], 0.5)
        self.assertEqual(report["selective_at"]["0.90"]["coverage"], 1.0)

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_pipeline(self, path=None, escalate=False):
        values = {"max_query_debate_rounds": 0, "aggregator": "vote"}
        if path is not None:
            values.update({"conformal_alpha": ALPHA, "conformal_model": path,
                           "conformal_escalate": escalate})
        generator = ScriptedGenerator(["The answer is: Paris", "The answer is: London"])
        pipeline = DragPlusPipeline(Flags(values), generator, FakeRetriever(),
                                    IdentityTemplate(), "NQ")
        item = RecordingItem("who wrote it?")
        pipeline.run([item])
        return item

    def test_default_and_single_element_set(self):
        default = self.run_pipeline()
        self.assertNotIn("conformal_set", default.output)
        answered = self.run_pipeline(write_calibration(self.temp.name, qhat=0.5, dataset="NQ"))
        self.assertFalse(answered.output["conformal_abstained"])
        self.assertEqual(answered.output["conformal_set"], ["paris"])

    def test_abstention_is_marked_pred_is_untouched_and_escalation_flags(self):
        item = self.run_pipeline(write_calibration(self.temp.name))
        self.assertEqual(item.pred, "Paris")
        self.assertTrue(item.output["conformal_abstained"])
        self.assertEqual(item.output["conformal_set"], ["london", "paris"])
        self.assertTrue(item.output["conformal_pooled"])
        escalated = self.run_pipeline(write_calibration(self.temp.name), True)
        self.assertTrue(escalated.output["conformal_escalated"])

    def test_flags_require_a_model_and_an_open_alpha(self):
        with self.assertRaises(FlagError):
            Flags({"conformal_alpha": ALPHA})
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha), self.assertRaises(FlagError):
                Flags({"conformal_alpha": alpha, "conformal_model": "x.json"})


class ScriptTests(unittest.TestCase):
    def test_record_distribution_validates_support_and_passages(self):
        record = {"dataset": "NQ", "gold": "Paris", "answers": ["Paris"],
                  "support": {"Paris": 0.9}}
        dataset, gold, scores = eval_conformal.record_distribution(record)
        self.assertEqual((dataset, gold), ("NQ", "paris"))
        self.assertAlmostEqual(sum(scores.values()), 1.0)
        with self.assertRaises(ValueError):
            eval_conformal.record_distribution({**record, "support": {"Paris": 2.0}})
        scored = eval_conformal.record_distribution(
            {**record, "passages": ["Paris is the capital"]})
        self.assertGreater(scored[2]["paris"], 0.0)

    def test_cross_fit_and_cli_write_a_report(self):
        records = [{"dataset": "NQ", "gold": "0", "answers": ["0", "1"],
                    "support": {"0": 0.5, "1": 0.5}} for _ in range(30)]
        path = Path(tempfile.mkdtemp()) / "records.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n",
                        encoding="utf-8")
        output = Path(tempfile.mkdtemp())
        code = eval_conformal.main(["--records", str(path), "--folds", "3",
                                    "--min-n", "5", "--out", str(output)])
        self.assertEqual(code, 0)
        self.assertTrue((output / "report-alpha-0.1.json").exists())
        self.assertTrue((output / "risk-coverage-0.1.svg").exists())


if __name__ == "__main__":
    unittest.main()
