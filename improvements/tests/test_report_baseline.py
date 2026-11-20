"""Pin the recorded baseline numbers (docs ledger, baseline snapshot) to the stored data."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "improvements"))
sys.path.insert(0, str(ROOT / "scripts"))
from drag_plus.analysis.report import (  # noqa: E402
    by_dataset, format_only_counts, load_pairs, pooled_summary, render)
import report_baseline  # noqa: E402

PAIRS_PATH = ROOT / "results" / "paired-200.json"


class BaselineReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = load_pairs(PAIRS_PATH)
        cls.summary = pooled_summary(cls.pairs)
        cls.rows = {row["dataset"]: row for row in by_dataset(cls.pairs)}

    def test_pooled_counts_and_p(self):
        self.assertEqual((self.summary["only_drag"], self.summary["only_naive"]), (24, 14))
        self.assertAlmostEqual(self.summary["p"], 0.1433, delta=1e-4)
        self.assertAlmostEqual(self.summary["naive_em"], 39.0)
        self.assertAlmostEqual(self.summary["drag_em"], 44.0)

    def test_per_dataset_p_values(self):
        self.assertAlmostEqual(self.rows["NQ"]["p"], 0.0625, delta=1e-4)
        self.assertAlmostEqual(self.rows["2wiki"]["p"], 0.0386, delta=1e-4)
        self.assertEqual(sum(row["n"] for row in self.rows.values()), 200)

    def test_holm_rejects_none(self):
        self.assertEqual(self.summary["holm_rejects"], [])

    def test_power_figures(self):
        self.assertAlmostEqual(self.summary["mde_pp"], 8.6, delta=0.2)
        self.assertAlmostEqual(self.summary["pairs_for_5pp"], 590, delta=15)

    def test_format_only_counts(self):
        counts = format_only_counts(self.pairs)
        self.assertEqual((counts["drag_wins_format_only"], counts["drag_wins"]), (5, 24))
        self.assertEqual((counts["naive_wins_format_only"], counts["naive_wins"]), (3, 14))

    def test_f1_bootstrap_is_seeded(self):
        again = pooled_summary(self.pairs)
        self.assertEqual(again["delta_f1_ci"], self.summary["delta_f1_ci"])
        low, high = self.summary["delta_f1_ci"]
        self.assertLess(low, self.summary["delta_f1"])
        self.assertLess(self.summary["delta_f1"], high)

    def test_script_prints_table(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(report_baseline.main(["--pairs", str(PAIRS_PATH)]), 0)
        text = out.getvalue()
        self.assertIn("| **Pooled** | 200 |", text)
        self.assertIn("Holm rejects: none", text)


def _pair(dataset, split, naive, drag, gold="Paris"):
    return {"dataset": dataset, "split": split, "id": f"{dataset}-{naive}-{drag}",
            "golden_answers": [gold], "naive_answer": naive, "drag_answer": drag}


class EdgeCaseReportTests(unittest.TestCase):
    def test_identical_arms_report_without_error(self):
        # An A/A run: zero discordant pairs, so Connor's power figures are undefined.
        pairs = [_pair("NQ", "dev", "Paris", "Paris"), _pair("NQ", "dev", "Rome", "Rome")]
        summary = pooled_summary(pairs)
        self.assertEqual(summary["p"], 1.0)
        self.assertIsNone(summary["pairs_for_5pp"])
        self.assertIsNone(summary["mde_pp"])
        self.assertIn("pairs for 5 pp: n/a", render(pairs))

    def test_unknown_dataset_is_kept_after_known_ones(self):
        pairs = [_pair("MuSiQue", "dev", "Rome", "Paris"), _pair("NQ", "dev", "Paris", "Paris")]
        rows = by_dataset(pairs)
        self.assertEqual([row["dataset"] for row in rows], ["NQ", "MuSiQue"])
        self.assertEqual(sum(row["n"] for row in rows), len(pairs))

    def test_mixed_splits_are_all_named(self):
        pairs = [_pair("PopQA", "test", "Paris", "Paris"), _pair("PopQA", "dev", "Paris", "Paris")]
        self.assertEqual(by_dataset(pairs)[0]["split"], "dev+test")



if __name__ == "__main__":
    unittest.main()
