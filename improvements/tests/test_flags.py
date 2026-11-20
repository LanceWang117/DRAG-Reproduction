"""Flag registry: defaults equal the baseline; bad input fails loudly."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.flags import FLAG_SPECS, FlagError, Flags, load_flags  # noqa: E402

BASELINE = {"max_query_debate_rounds": 3, "max_answer_debate_rounds": 3, "profile_calls": False,
            "answer_extractor": "native", "retrieval_cache": None}


class FlagDefaultTests(unittest.TestCase):
    def test_defaults_equal_the_baseline(self):
        self.assertEqual(Flags().to_manifest(), BASELINE)

    def test_every_spec_default_passes_its_own_validation(self):
        values = {spec.name: spec.default for spec in FLAG_SPECS}
        self.assertEqual(Flags(values), Flags())

    def test_attribute_access_and_immutability(self):
        flags = Flags({"max_query_debate_rounds": 1})
        self.assertEqual(flags.max_query_debate_rounds, 1)
        with self.assertRaises(AttributeError):
            flags.max_query_debate_rounds = 2
        with self.assertRaises(AttributeError):
            flags.not_a_flag

    def test_with_changes_returns_a_new_validated_object(self):
        changed = Flags().with_changes(answer_extractor="unified")
        self.assertEqual(changed.answer_extractor, "unified")
        self.assertEqual(Flags().answer_extractor, "native")
        with self.assertRaises(FlagError):
            Flags().with_changes(nope=1)


class FlagValidationTests(unittest.TestCase):
    def test_unknown_flag_is_an_error_naming_the_flag(self):
        with self.assertRaisesRegex(FlagError, "unknown flag.*max_query_rounds"):
            Flags({"max_query_rounds": 2})

    def test_wrong_type_is_an_error(self):
        for bad in ("3", 3.0, None, True):
            with self.subTest(value=bad), self.assertRaisesRegex(FlagError, "expects int"):
                Flags({"max_answer_debate_rounds": bad})

    def test_negative_rounds_and_bad_choice_are_errors(self):
        with self.assertRaisesRegex(FlagError, ">= 0"):
            Flags({"max_query_debate_rounds": -1})
        with self.assertRaisesRegex(FlagError, "must be one of"):
            Flags({"answer_extractor": "fancy"})

    def test_an_optional_flag_accepts_null_and_still_rejects_bad_values(self):
        self.assertIsNone(Flags({"retrieval_cache": None}).retrieval_cache)
        with self.assertRaisesRegex(FlagError, "expects str"):
            Flags({"retrieval_cache": 3})
        with self.assertRaisesRegex(FlagError, "must not be blank"):
            Flags({"retrieval_cache": "  "})


class FlagFileTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.addCleanup(Path(handle.name).unlink)
        handle.write(text)
        handle.close()
        return handle.name

    def test_load_from_json(self):
        path = self.write(json.dumps({"max_query_debate_rounds": 2}))
        self.assertEqual(load_flags(path).max_query_debate_rounds, 2)

    def test_load_rejects_unknown_flag_and_non_object(self):
        with self.assertRaises(FlagError):
            load_flags(self.write('{"bogus": 1}'))
        with self.assertRaisesRegex(FlagError, "JSON object"):
            load_flags(self.write("[1, 2]"))

    def test_manifest_round_trips_through_json(self):
        flags = Flags({"answer_extractor": "unified", "max_query_debate_rounds": 0})
        restored = Flags(json.loads(json.dumps(flags.to_manifest())))
        self.assertEqual(restored, flags)


if __name__ == "__main__":
    unittest.main()
