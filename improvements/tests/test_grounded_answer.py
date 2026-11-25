"""J18 grounded roles and verdict sequestration, on scripted outputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drag_plus import prompts  # noqa: E402
from drag_plus.aggregators import WITHHELD, sequester_verdict  # noqa: E402
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402

POOL = {"q": [{"id": "d1", "contents": "Paris\nParis is the capital of France."}]}
VERDICTS = [
    "Proponent reasoning.\nThe answer is: Paris",
    "Opponent reasoning.\nThe answer is: London",
    "The answer is: Paris",
    "Proponent round two.\nThe answer is: Paris",
    "Opponent round two.\nThe answer is: London",
    "The answer is: London",
]


def _content(prompt):
    return "\n".join(message["content"] for message in prompt)


def _run(outputs, docs=None, **flags):
    values = {"max_query_debate_rounds": 0, "max_answer_debate_rounds": 1}
    values.update(flags)
    generator = ScriptedGenerator(outputs)
    pipeline = DragPlusPipeline(Flags(values), generator,
                                FakeRetriever(POOL if docs is None else docs),
                                IdentityTemplate(), "NQ")
    item = RecordingItem("q")
    return pipeline.run_item(item), generator, item


class GroundedPromptTests(unittest.TestCase):
    def test_proponent_cites_labels_and_opponent_sees_the_evidence(self):
        pro = prompts.answer_agent_system("Proponent Agent 0", POOL, False, grounded=True)
        opp = prompts.answer_agent_system("Opponent Agent 0", POOL, False, grounded=True)
        self.assertIn("evidence synthesizer", pro["content"])
        self.assertIn("[E<query>-<rank>]", pro["content"])
        self.assertIn("E1-1", pro["content"])
        self.assertIn("Evidence:\nE1-1", opp["content"])
        self.assertIn("conflict auditor", opp["content"])
        self.assertNotIn("based on your own knowledge", opp["content"])

    def test_default_opponent_is_still_closed_book(self):
        opp = prompts.answer_agent_system("Opponent Agent 0", POOL, False)
        self.assertIn("based on your own knowledge", opp["content"])
        self.assertNotIn("E1-1", opp["content"])

    def test_grounded_pipeline_uses_grounded_prompts(self):
        state, generator, _ = _run(
            ["Proponent sees E1-1.\nThe answer is: Paris",
             "Conflict auditor sees E1-1.\nThe answer is: Paris",
             "The answer is: Paris"],
            opponent_mode="grounded")
        systems = [_content(call[0]) for call in generator.calls[:2]]
        self.assertTrue(any("evidence synthesizer" in system for system in systems))
        self.assertTrue(any("conflict auditor" in system for system in systems))
        self.assertTrue(all("E1-1" in system for system in systems))
        self.assertFalse(any("based on your own knowledge" in system
                             for system in systems))
        self.assertEqual(state.pred, "The answer is: Paris")

    def test_evidence_first_puts_one_labelled_prefix_on_both_agent_roles(self):
        _, generator, _ = _run(
            ["Proponent sees E1-1.\nThe answer is: Paris",
             "Auditor sees E1-1.\nThe answer is: Paris",
             "The answer is: Paris"],
            opponent_mode="grounded", prompt_layout="evidence_first")
        for call in generator.calls[:2]:
            system = _content(call[0])
            self.assertEqual(system.count("E1-1(Title:"), 1)
            self.assertIn("evidence is in the shared context above", system.lower())

    def test_sequestration_hides_peer_verdicts_but_keeps_reasoning(self):
        state, generator, _ = _run(VERDICTS, max_answer_debate_rounds=2,
                                   verdict_sequestration=True)
        first, second = (_content(call[0]) for call in generator.calls[3:5])
        self.assertIn("Opponent reasoning.", first)
        self.assertNotIn("The answer is: London", first)
        self.assertIn("Proponent reasoning.", second)
        self.assertNotIn("The answer is: Paris", second)
        self.assertIn("The answer is: ", first)
        self.assertIn("The answer is: Paris",
                      state.answer_messages["Proponent Agent 0"][2]["content"])

    def test_sequestration_off_keeps_the_peer_verdict(self):
        _, generator, _ = _run(VERDICTS, max_answer_debate_rounds=2)
        first, second = (_content(call[0]) for call in generator.calls[3:5])
        self.assertIn("The answer is: London", first)
        self.assertIn("The answer is: Paris", second)


class SequestrationTests(unittest.TestCase):
    def test_removes_every_marker_case_insensitively(self):
        text = "first\nThe answer is: Paris\nsecond\nTHE ANSWER IS: London\nthird"
        self.assertEqual(sequester_verdict(text), "first\n\nsecond\n\nthird")

    def test_inline_marker_keeps_the_reasoning(self):
        self.assertEqual(sequester_verdict("Reasoning. The answer is: Paris."),
                         "Reasoning.")

    def test_only_a_verdict_becomes_withheld(self):
        for text in ("The answer is: Paris", "THE ANSWER IS:", "  The answer is:  "):
            with self.subTest(text=text):
                self.assertEqual(sequester_verdict(text), WITHHELD)

    def test_text_without_a_marker_is_unchanged(self):
        self.assertEqual(sequester_verdict("reasoning only"), "reasoning only")


if __name__ == "__main__":
    unittest.main()
