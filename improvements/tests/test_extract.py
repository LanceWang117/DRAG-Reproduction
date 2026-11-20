"""Edge cases for the unified answer extractor E(.)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "improvements"))
from drag_plus.analysis.extract import extract_answer  # noqa: E402


class ExtractAnswerTests(unittest.TestCase):
    def test_text_after_marker(self):
        self.assertEqual(extract_answer("Some reasoning. The answer is: Paris."), "Paris")

    def test_last_marker_wins(self):
        text = "The answer is: Lyon. On reflection, the answer is: Paris"
        self.assertEqual(extract_answer(text), "Paris")

    def test_final_answer_marker_case_insensitive(self):
        self.assertEqual(extract_answer("FINAL ANSWER: 42"), "42")

    def test_only_first_line_after_marker(self):
        text = "The answer is: Dan Stevens\nHe voiced the Beast in the 2017 film."
        self.assertEqual(extract_answer(text), "Dan Stevens")

    def test_marker_with_nothing_after_uses_line_before(self):
        self.assertEqual(extract_answer("Nettie Fowler\nThe answer is:"), "Nettie Fowler")

    def test_marker_alone_gives_empty(self):
        self.assertEqual(extract_answer("The answer is:"), "")

    def test_no_marker_uses_first_non_empty_line(self):
        self.assertEqual(extract_answer("\n  October 13, 2017.\nMore text"), "October 13, 2017")

    def test_empty_and_none(self):
        self.assertEqual(extract_answer(""), "")
        self.assertEqual(extract_answer(None), "")

    def test_markdown_emphasis_and_trailing_punctuation(self):
        self.assertEqual(extract_answer("The answer is: **Ngaio Marsh**."), "Ngaio Marsh")

    def test_moderator_style_qualifiers_are_kept(self):
        # E(.) extracts the span; it never rewrites content (dev_8509 style).
        self.assertEqual(extract_answer("Dan Stevens (for singing scenes)"),
                         "Dan Stevens (for singing scenes)")

    def test_yes_no_normalisation(self):
        self.assertEqual(extract_answer("The answer is: yes, because", yes_no=True), "Yes")
        self.assertEqual(extract_answer("NO.", yes_no=True), "No")
        self.assertEqual(extract_answer("Nobody knows", yes_no=True), "Nobody knows")

    def test_yes_no_off_leaves_text(self):
        self.assertEqual(extract_answer("yes, because", yes_no=False), "yes, because")


if __name__ == "__main__":
    unittest.main()
