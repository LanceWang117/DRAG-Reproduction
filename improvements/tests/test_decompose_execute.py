"""J22 execution: grounded success, one rewrite, beam of two, call cap and runner arms."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.adapters import Backends, StageContext  # noqa: E402
from drag_plus.decompose.beam import Candidate, select_candidates
from drag_plus.decompose.execute import CallBudget, execute_plan
from drag_plus.decompose.plan import parse_plan
from drag_plus.decompose.route import run_decomposed
from drag_plus.flags import Flags
from drag_plus.pipeline import DragPlusPipeline
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.run import RunRequest, run_variant
from drag_plus.runner.variant import parse_variant
from drag_plus.state import QuestionState
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator

ORIGINAL = "Who was Junia Tertia's husband, and where did he die?"
HOP1 = "Who was Junia Tertia's husband?"
HOP2 = "Where did #E1 die?"
HOP2_CAESAR = "Where did Caesar die?"
PLAN = f"q1: {HOP1} -> #E1\nq2: {HOP2} -> #E2"


def context(retriever, outputs=(), flags=None):
    generator = outputs if hasattr(outputs, "generate") else ScriptedGenerator(list(outputs))
    return StageContext(Flags(flags or {}),
                        Backends(generator, retriever, IdentityTemplate()), "NQ")


def answers(mapping):
    return lambda ctx, state, query, passages: mapping[query]


def fixed(entity, confidence=1.0):
    return lambda question, answer, passages: [Candidate(entity, confidence)]


def docs():
    return {HOP1: [{"id": "d1", "contents": "Junia Tertia married Caesar."}],
            HOP2_CAESAR: [{"id": "d2", "contents": "Caesar died in Rome."}]}


class GroundedTests(unittest.TestCase):
    def test_grounded_path_binds_and_composes(self):
        retriever = FakeRetriever(docs())
        state = QuestionState(ORIGINAL)

        def candidates(question, answer, passages):
            return [Candidate(answer.split("The answer is: ")[-1].strip(), 1.0)]

        result = execute_plan(context(retriever), state, parse_plan(PLAN),
                              answer_fn=answers({HOP1: "The answer is: Caesar",
                                                 HOP2_CAESAR: "The answer is: Rome"}),
                              candidate_fn=candidates)
        self.assertEqual([row.entity for row in result.ledger], ["Caesar", "Rome"])
        self.assertEqual([row.doc_ids for row in result.ledger], [("d1",), ("d2",)])
        self.assertEqual(result.composition.answer, "Rome")
        self.assertIn("The answer is: Rome", result.composition.text)
        self.assertEqual(result.calls, len(retriever.calls) + 2)  # searches + 2 answers
        self.assertEqual(state.evidence.ids(), {"d1", "d2", f"doc::{ORIGINAL}"})

    def test_grounding_failure_gets_exactly_one_rewrite(self):
        mapping = docs()
        mapping[HOP1] = [{"id": "d1a", "contents": "No husband is named here."}]
        mapping[HOP1 + " evidence"] = [{"id": "d1b", "contents": "Caesar was the husband."}]
        mapping[HOP2_CAESAR] = [{"id": "d2", "contents": "Caesar died in Rome."}]
        retriever = FakeRetriever(mapping)
        state = QuestionState(ORIGINAL)
        result = execute_plan(context(retriever), state, parse_plan(PLAN),
                              answer_fn=answers({HOP1: "The answer is: Caesar",
                                                 HOP1 + " evidence": "The answer is: Caesar",
                                                 HOP2_CAESAR: "The answer is: Rome"}),
                              candidate_fn=fixed("Caesar"))
        self.assertIsNotNone(result)
        self.assertEqual(retriever.calls[:3], [ORIGINAL, HOP1, HOP1 + " evidence"])

    def test_failed_rewrite_falls_back_without_composing(self):
        retriever = FakeRetriever({})
        state = QuestionState(ORIGINAL)
        result = execute_plan(context(retriever), state, parse_plan(PLAN),
                              answer_fn=answers({HOP1: "The answer is: Caesar",
                                                 HOP1 + " evidence": "The answer is: Caesar"}),
                              candidate_fn=fixed("Caesar"))
        self.assertIsNone(result)
        self.assertEqual(retriever.calls, [ORIGINAL, HOP1, HOP1 + " evidence"])

    def test_call_cap_stops_retrieval_and_marks_the_budget(self):
        budget = CallBudget(2)
        retriever = FakeRetriever({})
        state = QuestionState(ORIGINAL)
        plan = parse_plan("q1: one -> #E1\nq2: uses #E1 -> #E2\nq3: uses #E2 -> #E3")
        result = execute_plan(context(retriever), state, plan, budget=budget,
                              answer_fn=answers({}), candidate_fn=fixed("x"))
        self.assertIsNone(result)
        self.assertEqual((budget.used, budget.limit), (2, 2))
        self.assertTrue(budget.exhausted)


class BeamTests(unittest.TestCase):
    def test_low_margin_keeps_two_candidates_and_high_margin_keeps_one(self):
        candidates = [Candidate("Alpha", 0.60), Candidate("Beta", 0.55)]
        self.assertEqual([c.entity for c in select_candidates(candidates)], ["Alpha", "Beta"])
        wide = [Candidate("Alpha", 0.90), Candidate("Beta", 0.50)]
        self.assertEqual([c.entity for c in select_candidates(wide)], ["Alpha"])

    def test_execution_keeps_two_beams_through_the_next_hop(self):
        mapping = {HOP1: [{"id": "a", "contents": "Alpha and Beta were both mentioned."}],
                   "Where did Alpha live?": [{"id": "a2", "contents": "Alpha lived in Rome."}],
                   "Where did Beta live?": [{"id": "b2", "contents": "Beta lived in Troy."}]}
        retriever = FakeRetriever(mapping)
        state = QuestionState(ORIGINAL)
        answers_map = {HOP1: "The answer is: Alpha", "Where did Alpha live?": "The answer is: Rome",
                       "Where did Beta live?": "The answer is: Troy"}

        def candidates(question, answer, passages):
            if question == HOP1:
                return [Candidate("Alpha", 0.60), Candidate("Beta", 0.55)]
            entity = answer.split("The answer is: ")[-1].strip()
            return [Candidate(entity, 1.0)]

        plan = parse_plan(f"q1: {HOP1} -> #E1\nq2: Where did #E1 live? -> #E2")
        result = execute_plan(context(retriever), state, plan,
                              answer_fn=answers(answers_map), candidate_fn=candidates)
        self.assertEqual(result.beams, 2)
        self.assertEqual(result.ledger[0].entity, "Alpha")
        self.assertEqual([c for c in retriever.calls if "Alpha" in c or "Beta" in c],
                         ["Where did Alpha live?", "Where did Beta live?"])


class RouteTests(unittest.TestCase):
    def test_pipeline_runs_the_route_when_the_flag_is_on(self):
        retriever = FakeRetriever({ORIGINAL: [{"id": "o", "contents": "Junia Tertia."}],
                                   **docs()})
        generator = ScriptedGenerator([PLAN, "The answer is: Caesar", "The answer is: Rome"])
        pipeline = DragPlusPipeline(Flags({"decompose": True}), generator, retriever,
                                    IdentityTemplate(), "NQ")
        item = RecordingItem(ORIGINAL)
        state = pipeline.run_item(item)
        self.assertEqual(state.pred, "Rome")
        self.assertEqual(item.output["pred"], "Rome")
        keys = [key for key, _ in state.outputs]
        self.assertIn("Decompose_Plan", keys)
        self.assertIn("Decompose_Ledger", keys)
        self.assertIn("The answer is: Rome", state.raw_pred)

    def test_invalid_plan_falls_back_before_any_retrieval(self):
        retriever = FakeRetriever({})
        generator = ScriptedGenerator(["not a plan"])
        state = QuestionState(ORIGINAL)
        self.assertFalse(run_decomposed(context(retriever, generator, {"decompose": True}),
                                        state))
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(retriever.calls, [])
        self.assertEqual(state.decisions[0]["decision"], "fallback")

    def test_ungrounded_route_resets_the_partial_pool(self):
        retriever = FakeRetriever({})
        generator = ScriptedGenerator([PLAN, "The answer is: Caesar", "The answer is: Caesar"])
        state = QuestionState(ORIGINAL)
        self.assertFalse(run_decomposed(context(retriever, generator, {"decompose": True}),
                                        state))
        self.assertIsNone(state.evidence)
        self.assertEqual(state.query_pool, {})


class ComparisonArmTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_ircot_and_selfask_run_through_the_runner_with_the_fake_backend(self):
        variant = parse_variant({"name": "cmp", "arms": ["ircot", "selfask"]})
        rows = [{"dataset": "NQ", "split": "dev", "id": "q0",
                 "question": "who wrote it?", "golden_answers": ["answer 1"]}]
        request = RunRequest(rows=rows, variant=variant, backend=build_fake_backend(),
                             out_dir=self.root / "run", environment=lambda: {"commit": "t"})
        manifest = run_variant(request)
        self.assertEqual(manifest["exit_status"], "complete")
        lines = [json.loads(line) for line in
                 (self.root / "run" / "transcripts.jsonl").read_text().splitlines()]
        self.assertEqual({row["arm"] for row in lines}, {"ircot", "selfask"})
        self.assertTrue(all(row["pred"] for row in lines))


if __name__ == "__main__":
    unittest.main()
