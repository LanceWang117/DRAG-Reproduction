"""J19 faithfulness scorers: deterministic heuristic, lazy MiniCheck, no hidden import."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "improvements"))

from drag_plus.grounding.minicheck import MiniCheckFaithfulness  # noqa: E402
from drag_plus.grounding.scorer import (FaithfulnessConfigError,  # noqa: E402
                                        HeuristicFaithfulness, build_scorer)


class HeuristicTests(unittest.TestCase):
    def test_present_and_absent_answers(self):
        scorer = HeuristicFaithfulness()
        self.assertEqual(scorer.score("Paris", ["Paris is the capital of France."]), 1.0)
        self.assertEqual(scorer.score("Berlin", ["Paris is the capital of France."]), 0.0)

    def test_normalisation_case_and_markdown(self):
        scorer = HeuristicFaithfulness()
        self.assertEqual(scorer.score("The answer is: **Paris**.", ["paris, France"]), 1.0)
        self.assertEqual(scorer.score("  New   York ", ["New York City"]), 1.0)

    def test_whole_word_not_substring(self):
        self.assertEqual(HeuristicFaithfulness().score("Paris", ["A Parisian cafe"]), 0.0)

    def test_fraction_of_cited_passages(self):
        passages = ["Paris is in France.", "London is in England.", "Paris, Texas"]
        self.assertAlmostEqual(HeuristicFaithfulness().score("Paris", passages), 2 / 3)

    def test_yes_no_is_unknown_not_perfect(self):
        scorer = HeuristicFaithfulness(yes_no=True)
        self.assertEqual(scorer.score("The answer is: Yes, because it is.", ["Yes"]), 0.5)
        self.assertEqual(HeuristicFaithfulness().score("No", ["No"]), 0.5)

    def test_empty_answer_or_no_passages_scores_zero(self):
        scorer = HeuristicFaithfulness()
        self.assertEqual(scorer.score("", ["Paris"]), 0.0)
        self.assertEqual(scorer.score("Paris", []), 0.0)
        self.assertEqual(scorer.score("Paris", ["   "]), 0.0)

    def test_build_scorer_default_and_unknown(self):
        self.assertIsInstance(build_scorer(None), HeuristicFaithfulness)
        self.assertIsInstance(build_scorer("heuristic"), HeuristicFaithfulness)
        with self.assertRaisesRegex(FaithfulnessConfigError, "unknown faithfulness_scorer"):
            build_scorer("nli")


class FakeMiniCheck:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.calls = []

    def score(self, docs, claims):
        self.calls.append((list(docs), list(claims)))
        return [1] * len(docs), list(self.probabilities), docs, []


class MiniCheckAdapterTests(unittest.TestCase):
    def test_scores_each_passage_and_takes_the_maximum(self):
        fake = FakeMiniCheck([0.2, 0.9])
        scorer = MiniCheckFaithfulness(loader=lambda name, cache: fake)
        value = scorer.score("The answer is: **Paris**.", ["passage one", "passage two"])
        self.assertEqual(value, 0.9)
        self.assertEqual(fake.calls, [(["passage one", "passage two"], ["Paris", "Paris"])])

    def test_import_error_is_a_clear_configuration_error(self):
        def broken(name, cache):
            raise ImportError("minicheck is not installed")
        scorer = MiniCheckFaithfulness(loader=broken)
        with self.assertRaisesRegex(FaithfulnessConfigError, "minicheck.*install"):
            scorer.score("Paris", ["Paris"])

    def test_empty_input_never_loads_the_model(self):
        def explode(name, cache):
            raise AssertionError("model must not load for empty input")
        self.assertEqual(MiniCheckFaithfulness(loader=explode).score("", ["Paris"]), 0.0)
        self.assertEqual(MiniCheckFaithfulness(loader=explode).score("Paris", []), 0.0)

    def test_default_path_never_imports_minicheck(self):
        code = ("import sys\n"
                f"sys.path.insert(0, {str(ROOT / 'improvements')!r})\n"
                "import drag_plus.grounding as grounding\n"
                "assert grounding.build_scorer(None).score('Paris', ['Paris']) == 1.0\n"
                "assert 'minicheck' not in sys.modules\n")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
