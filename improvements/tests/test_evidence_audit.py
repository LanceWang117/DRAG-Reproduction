"""Regex rewrite audit: it flags introduced entities/numbers and never blocks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.evidence import audit_rewrite, extract_entities, extract_numbers  # noqa: E402


class AuditTests(unittest.TestCase):
    def test_numbers_introduced_by_a_rewrite_are_flagged_and_existing_ones_are_not(self):
        self.assertEqual(audit_rewrite("from 1964 to 1985", "from 1964 to 1985")
                         ["invented_numbers"], [])
        result = audit_rewrite("who wrote it", "who wrote the novel in 1985")
        self.assertEqual(result["invented_numbers"], ["1985"])
        self.assertEqual(extract_numbers("1,200") , extract_numbers("1200"))
        self.assertEqual(extract_numbers("40%"), {"40%"})

    def test_entities_use_capitalisation_heuristics_mid_sentence(self):
        result = audit_rewrite("who wrote it", "who wrote the novel by George Orwell")
        self.assertIn("george orwell", result["invented_entities"])
        self.assertNotIn("who", extract_entities("Who wrote it?"))
        self.assertEqual(extract_entities("The novel") & {"the"}, set())

    def test_audit_is_a_pure_empty_result_when_nothing_is_invented(self):
        self.assertEqual(audit_rewrite("Jaffa Cakes in 1991", "Jaffa Cakes in 1991"),
                         {"invented_entities": [], "invented_numbers": []})


if __name__ == "__main__":
    unittest.main()
