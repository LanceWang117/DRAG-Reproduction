"""The scorer must reproduce FlashRAG's stored per-question metrics exactly."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "improvements"))
from drag_plus.analysis.score import (  # noqa: E402
    exact_match, normalize_answer, score_answer, substring_match, token_scores)

METRICS = ("em", "acc", "f1", "precision", "recall")


class NormalizeTests(unittest.TestCase):
    def test_lowercases_and_strips_punctuation_and_articles(self):
        self.assertEqual(normalize_answer("The  Beatles!"), "beatles")
        self.assertEqual(normalize_answer("A, an, THE end."), "end")

    def test_empty_string(self):
        self.assertEqual(normalize_answer(""), "")


class MetricTests(unittest.TestCase):
    def test_exact_match_any_gold(self):
        self.assertEqual(exact_match("Pelé.", ["Edson", "pelé"]), 1.0)
        self.assertEqual(exact_match("Romário", ["Pelé"]), 0.0)

    def test_gold_may_be_a_string(self):
        self.assertEqual(exact_match("Paris", "paris"), 1.0)

    def test_substring_match(self):
        self.assertEqual(substring_match("Dan Stevens (for singing scenes)", ["Dan Stevens"]), 1.0)
        self.assertEqual(exact_match("Dan Stevens (for singing scenes)", ["Dan Stevens"]), 0.0)

    def test_token_scores_take_maximum_over_golds(self):
        scores = token_scores("October 13, 2017, to April 20, 2018.", ["October 13, 2017"])
        self.assertAlmostEqual(scores["recall"], 1.0)
        self.assertAlmostEqual(scores["precision"], 3 / 7)

    def test_yes_no_mismatch_scores_zero(self):
        self.assertEqual(token_scores("yes", ["no"])["f1"], 0)
        self.assertEqual(token_scores("no it is", ["no"])["f1"], 0)

    def test_empty_prediction_scores_zero_without_error(self):
        self.assertEqual(score_answer("", ["Paris"])["f1"], 0)


class StoredMetricsRegressionTests(unittest.TestCase):
    """All 400 stored answers (200 questions x 2 methods) must re-score exactly."""

    @classmethod
    def setUpClass(cls):
        path = ROOT / "results" / "paired-200.json"
        cls.pairs = json.loads(path.read_text())["pairs"]

    def test_all_stored_metrics_match(self):
        checked = 0
        for pair in self.pairs:
            for method in ("naive", "drag"):
                got = score_answer(pair[f"{method}_answer"], pair["golden_answers"])
                stored = pair[f"{method}_metrics"]
                for metric in METRICS:
                    self.assertAlmostEqual(got[metric], stored[metric], places=12,
                                           msg=f"{pair['id']} {method} {metric}")
                checked += 1
        self.assertEqual(checked, 400)


if __name__ == "__main__":
    unittest.main()
