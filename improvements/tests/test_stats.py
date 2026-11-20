"""Known values and edge cases for the paired statistics."""
from math import comb
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "improvements"))
from drag_plus.analysis.stats import (  # noqa: E402
    discordant_counts, holm, mcnemar_p, minimum_detectable_effect,
    paired_bootstrap_ci, required_pairs, tost_paired, wilcoxon_signed_rank)


class McNemarTests(unittest.TestCase):
    def test_recorded_baseline_values(self):
        self.assertAlmostEqual(mcnemar_p(24, 14), 0.1433, places=4)
        self.assertAlmostEqual(mcnemar_p(0, 5), 0.0625, places=10)
        self.assertAlmostEqual(mcnemar_p(10, 2), 0.0386, places=4)

    def test_symmetric_in_b_and_c(self):
        self.assertEqual(mcnemar_p(3, 9), mcnemar_p(9, 3))

    def test_no_discordant_pairs_or_equal_counts_give_one(self):
        self.assertEqual(mcnemar_p(0, 0), 1.0)
        self.assertEqual(mcnemar_p(3, 3), 1.0)
        self.assertEqual(mcnemar_p(3, 3, midp=True), 1.0)

    def test_midp_is_exact_minus_point_probability(self):
        exact, mid = mcnemar_p(10, 2), mcnemar_p(10, 2, midp=True)
        self.assertAlmostEqual(exact - mid, comb(12, 2) / 2 ** 12, places=12)

    def test_large_n_does_not_overflow(self):
        self.assertLess(mcnemar_p(700, 500), 1e-6)

    def test_discordant_counts(self):
        self.assertEqual(discordant_counts([1, 1, 0, 0], [0, 1, 1, 0]), (1, 1))


class HolmTests(unittest.TestCase):
    def test_step_down(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.005]), [True, False, False, True])

    def test_recorded_baseline_rejects_none(self):
        p_values = [0.0625, 1.0, 0.2188, 0.0386, 0.7266, 1.0]
        self.assertEqual(holm(p_values), [False] * 6)


class SampleSizeTests(unittest.TestCase):
    def test_required_pairs_for_five_points(self):
        self.assertTrue(575 <= required_pairs(0.19, 0.05) <= 605)

    def test_mde_at_two_hundred(self):
        self.assertAlmostEqual(minimum_detectable_effect(200, 0.19), 0.086, delta=0.002)

    def test_mde_and_required_pairs_agree(self):
        delta = minimum_detectable_effect(400, 0.19)
        self.assertIn(required_pairs(0.19, delta), (400, 401))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            required_pairs(0.19, 0.5)


class BootstrapTests(unittest.TestCase):
    def test_deterministic_for_seed(self):
        diffs = [1, 0, 0, -1, 1, 1, 0, 0, 1, 0] * 5
        self.assertEqual(paired_bootstrap_ci(diffs, seed=7), paired_bootstrap_ci(diffs, seed=7))

    def test_interval_contains_mean(self):
        mean, low, high = paired_bootstrap_ci([0.2, 0.1, 0.0, 0.3, -0.1] * 8, seed=1)
        self.assertLessEqual(low, mean)
        self.assertLessEqual(mean, high)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([])


class EquivalenceTests(unittest.TestCase):
    def test_identical_arms_are_equivalent(self):
        self.assertTrue(tost_paired([0.0] * 60, margin=0.005)["equivalent"])

    def test_clear_difference_is_not_equivalent(self):
        self.assertFalse(tost_paired([1, 0, 1, 1, 0, 1] * 10, margin=0.01)["equivalent"])


class WilcoxonTests(unittest.TestCase):
    def test_all_zero_differences(self):
        self.assertEqual(wilcoxon_signed_rank([0, 0, 0])["p"], 1.0)

    def test_consistent_shift_is_significant(self):
        result = wilcoxon_signed_rank([-2.5, -1.0, -3.2, -0.8, -1.9] * 6)
        self.assertLess(result["p"], 0.001)
        self.assertEqual(result["w_plus"], 0.0)

    def test_float_ties_are_grouped(self):
        # 2/3 and 1 - 1/3 differ in the last ulp but are the same F1 difference.
        float_ties = wilcoxon_signed_rank([2 / 3, 1 - 1 / 3, -0.5, 0.25, 0.8])
        exact_ties = wilcoxon_signed_rank([2 / 3, 2 / 3, -0.5, 0.25, 0.8])
        self.assertEqual(float_ties, exact_ties)

    def test_near_zero_differences_are_dropped(self):
        self.assertEqual(wilcoxon_signed_rank([1e-17, 1.0, 2.0])["n"], 2)

    def test_symmetric_differences_are_not_significant(self):
        self.assertGreater(wilcoxon_signed_rank([1, -1, 2, -2, 3, -3])["p"], 0.5)


if __name__ == "__main__":
    unittest.main()
