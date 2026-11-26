"""J20: fitting per-route correctness models and the Pareto evaluation against random.

The synthetic data has a known signal: each route's correctness depends on a different
pre-debate feature, so a fitted router must beat equal-share random routing. The tests
also cover the test-split refusal, cost averaging, LODO output, and the SVG figure.
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
import eval_router  # noqa: E402
import fit_router  # noqa: E402
from drag_plus.gates.router import ROUTE_NAMES, load_router  # noqa: E402

DATASETS = ("NQ", "HotpotQA", "2Wiki")


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def synthetic(count=300, seed=11):
    """Each route has its own signal: margin, s_bar_O or agreement+entropy."""
    rng = random.Random(seed)
    records = []
    for index in range(count):
        dataset = DATASETS[index % len(DATASETS)]
        features = {"retrieval_margin": rng.random(), "s_bar_O": rng.random(),
                    "answer_agreement": rng.random(), "naive_entropy": rng.random()}
        probabilities = {"naive": _sigmoid(4.0 * features["retrieval_margin"] - 2.2),
                         "refine_only": _sigmoid(4.0 * features["s_bar_O"] - 2.2),
                         "full_drag": _sigmoid(3.0 * features["answer_agreement"]
                                               + 3.0 * features["naive_entropy"] - 2.4)}
        for route in ROUTE_NAMES:
            records.append({"question_id": f"q{index}", "dataset": dataset,
                            "route": route, "features": dict(features),
                            "correct": rng.random() < probabilities[route],
                            "cost": {"naive": 1.0, "refine_only": 4.0,
                                     "full_drag": 10.0}[route]})
    return records


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8")
    return path


def fit(directory, records, *extra):
    source = write_jsonl(Path(directory) / "records.jsonl", records)
    out = Path(directory) / "router.json"
    code = fit_router.main(["--input", str(source), "--out", str(out), *extra])
    return code, out


class FitScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_synthetic_signal_fits_all_routes_with_lodo(self):
        code, path = fit(self.temp.name, synthetic())
        self.assertEqual(code, 0)
        model = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(model["routes"]), set(ROUTE_NAMES))
        self.assertEqual(model["feature_names"], list(eval_router.ROUTER_FEATURE_NAMES))
        self.assertEqual(model["training_split"], "train")
        self.assertEqual(len(model["manifest_hash"]), 64)
        self.assertEqual(set(model["metrics"]["lodo_auroc"]), set(DATASETS))
        self.assertGreater(model["metrics"]["lodo_macro_auroc"], 0.55)

    def test_test_split_is_refused_before_any_output(self):
        code, path = fit(self.temp.name, synthetic(), "--split", "test")
        self.assertEqual(code, 2)
        self.assertFalse(path.exists())

    def test_recorded_costs_are_averaged_and_missing_features_rejected(self):
        records = synthetic(30)
        next(record for record in records if record["route"] == "full_drag")["cost"] = 99.0
        code, path = fit(self.temp.name, records)
        self.assertEqual(code, 0)
        model = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreater(model["costs"]["full_drag"], 10.0)
        with self.assertRaises(ValueError):
            fit_router._features({"retrieval_margin": 0.0})

    def test_unknown_route_and_missing_label_are_rejected(self):
        record = {"question_id": "q0", "route": "magic",
                  "features": {name: 0.0 for name in eval_router.ROUTER_FEATURE_NAMES},
                  "correct": True}
        with self.assertRaises(ValueError):
            fit_router._route(record)
        with self.assertRaises(ValueError):
            fit_router._correct({"route": "naive"})


class EvalScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.records = synthetic()
        code, self.model_path = fit(self.temp.name, self.records)
        self.assertEqual(code, 0)
        self.model = load_router(self.model_path)

    def test_router_beats_equal_share_random_on_accuracy(self):
        result = eval_router.evaluate(self.records, self.model,
                                      [0.0, 0.02, 0.05], seeds=100, seed=0)
        self.assertEqual(len(result["router"]), 3)
        advantage = max(chosen["accuracy"] - baseline["accuracy_mean"]
                        for chosen, baseline in zip(result["router"], result["random"]))
        self.assertGreater(advantage, 0.02)
        for baseline in result["random"]:
            self.assertLessEqual(baseline["accuracy_low"], baseline["accuracy_mean"])
            self.assertGreaterEqual(baseline["accuracy_high"], baseline["accuracy_mean"])

    def test_main_writes_a_pareto_svg(self):
        figure = Path(self.temp.name) / "pareto.svg"
        code = eval_router.main(["--input", str(Path(self.temp.name) / "records.jsonl"),
                                 "--model", str(self.model_path), "--figure", str(figure),
                                 "--seeds", "50", "--lambdas", "0,0.02,0.05"])
        self.assertEqual(code, 0)
        svg = figure.read_text(encoding="utf-8")
        self.assertIn("<svg", svg)
        self.assertIn("Pareto", svg)
        self.assertIn("polyline", svg)
        self.assertIn("circle", svg)

    def test_grouping_requires_every_route(self):
        with self.assertRaisesRegex(ValueError, "all routes"):
            eval_router.group_questions(self.records[:-1], self.model)


if __name__ == "__main__":
    unittest.main()
