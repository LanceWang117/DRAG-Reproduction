"""J22 plan parser/validator: every invalid-plan type from the spec."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.decompose.plan import (MAX_STEPS, Plan, PlanError, SubQuestion, parse_plan,  # noqa: E402
                                      planner_messages, single_hop_plan, validate_plan)

VALID = "q1: Who was Junia Tertia's husband? -> #E1\nq2: Where did #E1 die? -> #E2"


class ParseTests(unittest.TestCase):
    def test_valid_plan_binds_backward_placeholders(self):
        plan = parse_plan(VALID + "\n")
        self.assertEqual([s.qid for s in plan.steps], [1, 2])
        self.assertEqual(plan.steps[0].output, "#E1")
        self.assertIn("#E1", plan.steps[1].text)
        self.assertEqual(plan.max_depth, 2)

    def test_leading_plan_header_is_ignored(self):
        self.assertEqual(parse_plan("Plan:\n" + VALID).max_depth, 2)

    def test_empty_plan_is_rejected(self):
        for raw in ("", "   \n", None, 3):
            with self.subTest(raw=raw), self.assertRaises(PlanError):
                parse_plan(raw)

    def test_depth_above_three_is_rejected(self):
        raw = "\n".join(f"q{i}: hop {i} -> #E{i}" for i in range(1, MAX_STEPS + 2))
        with self.assertRaises(PlanError) as caught:
            parse_plan(raw)
        self.assertIn("depth", str(caught.exception))

    def test_forward_reference_is_a_cycle(self):
        with self.assertRaises(PlanError) as caught:
            parse_plan("q1: uses #E2 -> #E1\nq2: second -> #E2")
        self.assertIn("cycles", str(caught.exception))

    def test_self_reference_is_a_cycle(self):
        with self.assertRaises(PlanError) as caught:
            parse_plan("q1: uses #E1 -> #E1")
        self.assertIn("cycles", str(caught.exception))

    def test_undefined_placeholder_is_rejected(self):
        with self.assertRaises(PlanError) as caught:
            parse_plan("q1: first -> #E1\nq2: uses #E0 -> #E2")
        self.assertIn("undefined", str(caught.exception))

    def test_duplicate_qid_is_rejected(self):
        with self.assertRaises(PlanError):
            parse_plan("q1: first -> #E1\nq1: second -> #E1")

    def test_wrong_output_placeholder_is_rejected(self):
        with self.assertRaises(PlanError) as caught:
            parse_plan("q1: first -> #E2")
        self.assertIn("binds", str(caught.exception))

    def test_empty_sub_question_is_rejected(self):
        with self.assertRaises(PlanError):
            parse_plan("q1:    -> #E1")

    def test_malformed_line_is_rejected(self):
        for raw in ("who did it", "q1 first -> #E1", "q1: first #E1", "Q1: first -> #E1"):
            with self.subTest(raw=raw), self.assertRaises(PlanError):
                parse_plan(raw)

    def test_planner_prompt_names_the_question_and_the_grammar(self):
        messages = planner_messages("Who was Junia Tertia's husband?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("Junia Tertia", messages[1]["content"])
        self.assertIn("#E1", messages[0]["content"])
        self.assertIn("at most three", messages[0]["content"])


class ValidationTests(unittest.TestCase):
    def test_single_hop_plan_is_valid_and_executable(self):
        plan = single_hop_plan("Who wrote it?")
        self.assertEqual(validate_plan(plan), plan)
        self.assertEqual(plan.steps[0].text, "Who wrote it?")

    def test_validate_rejects_a_plan_out_of_order(self):
        with self.assertRaises(PlanError):
            validate_plan(Plan((SubQuestion(2, "second", "#E2"),)))

    def test_validate_rejects_a_non_step(self):
        with self.assertRaises(PlanError):
            validate_plan(Plan(("q1",)))

    def test_validate_rejects_an_empty_plan_object(self):
        with self.assertRaises(PlanError):
            validate_plan(Plan(()))


if __name__ == "__main__":
    unittest.main()
