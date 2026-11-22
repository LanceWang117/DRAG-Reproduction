"""Structured Opponent actions and Judge visibility (J13).

Covers every parser rejection reason, code fences/trailing text, the stable-ID
regression against the Levenshtein near-tie, transactional rollback, and both flag
states through the pipeline.
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.adapters import Backends, StageContext  # noqa: E402
from drag_plus.evidence.actions import (ACTION_APPLIED, EXIT_ZERO_NOVELTY,  # noqa: E402
                                        find_most_similar_key)
from drag_plus.evidence.store import register_query  # noqa: E402
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.judge.action_schema import (ActionError, parse_action,  # noqa: E402
                                           validate_action)
from drag_plus.judge.actions import apply_opponent_action  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from drag_plus.state import QuestionState  # noqa: E402

from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402
from t0_scenarios import answer_rounds, query_round  # noqa: E402


def raw(**overrides):
    data = {"action": "expand", "source_query_id": 1,
            "new_query": "a better query", "reason": "it is clearer"}
    data.update(overrides)
    return json.dumps(data)


def doc(doc_id, body="body"):
    return {"id": str(doc_id), "contents": f"Title {doc_id}\n{body}"}


def state_with(*queries):
    """A state with the given active queries registered in first-seen order."""
    state = QuestionState("original question?")
    state.query_pool = {query: [doc(f"d{index}")] for index, query in
                        enumerate(queries, start=1)}
    for index, query in enumerate(queries, start=1):
        register_query(state.queries, query,
                       "original" if index == 1 else "expand", index - 1)
    return state


def context(flags=None, retriever=None):
    retriever = retriever or FakeRetriever()
    return StageContext(Flags(flags or {}), Backends(
        ScriptedGenerator([]), retriever, IdentityTemplate()), "NQ")


class ParserTests(unittest.TestCase):
    def test_takes_the_last_object_after_a_fence_and_trailing_text(self):
        text = "reasoning\n```json\n" + raw() + "\n```\ntrailing words"
        self.assertEqual(parse_action(text).action, "expand")
        first = json.dumps({"note": "an example"})
        self.assertEqual(parse_action(first + "\n" + raw(action="optimize")).action,
                         "optimize")

    def test_rejects_multiple_actions_unknown_and_missing_fields(self):
        cases = [
            (raw() + " " + raw(), "multiple action"),
            (raw(extra=1), "unknown fields"),
            (json.dumps({"action": "no_op"}), "missing fields"),
            ("no json here", "no JSON object"),
            (raw() + "\n{\"action\":", "unclosed JSON object"),
        ]
        for text, message in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ActionError, message):
                parse_action(text)

    def test_rejects_wrong_types_and_empty_queries(self):
        cases = [
            (raw(action=1), "action must be"),
            (raw(source_query_id="1"), "source_query_id must be"),
            (raw(source_query_id=True), "source_query_id must be"),
            (raw(new_query=3), "new_query must be"),
            (raw(reason=3), "reason must be"),
            (raw(new_query="  "), "new_query must not be empty"),
        ]
        for text, message in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ActionError, message):
                parse_action(text)

    def test_no_op_accepts_empty_query_and_missing_source(self):
        action = parse_action(raw(action="no_op", source_query_id=None, new_query=""))
        self.assertEqual((action.action, action.new_query), ("no_op", ""))


class ValidationTests(unittest.TestCase):
    def test_optimize_needs_an_active_source_id(self):
        state = state_with("who wrote Hamlet?", "who wrote the play now")
        for source, message in ((None, "needs a source_query_id"),
                                (99, "unknown source_query_id")):
            text = raw(action="optimize", source_query_id=source,
                       new_query="who wrote the novel", reason="clarify")
            with self.subTest(source=source), self.assertRaisesRegex(ActionError, message):
                validate_action(parse_action(text), state)

    def test_length_bounds_and_duplicate_and_entity_rules(self):
        state = state_with("who wrote Hamlet?", "who wrote the play now")
        for query, message in (("two words", "tokens"),
                               (" ".join(["word"] * 31), "tokens"),
                               ("who wrote Hamlet", "duplicates")):
            text = raw(action="expand", new_query=query, reason="clarify")
            with self.subTest(query=query), self.assertRaisesRegex(ActionError, message):
                validate_action(parse_action(text), state)
        for query in ("one two three", " ".join(["word"] * 30)):
            text = raw(action="expand", new_query=query, reason="clarify")
            validate_action(parse_action(text), state)
        dropped = raw(new_query="who wrote the play", reason="")
        with self.assertRaisesRegex(ActionError, "drops entities"):
            validate_action(parse_action(dropped), state)
        kept = raw(new_query="who wrote the play", reason="drop Hamlet")
        self.assertEqual(validate_action(parse_action(kept), state), "who wrote Hamlet?")

    def test_expand_without_source_uses_the_latest_refined_query(self):
        state = state_with("who wrote Hamlet?", "who wrote the play now")
        action = parse_action(raw(action="expand", source_query_id=None,
                                  new_query="a fresh new query"))
        self.assertEqual(validate_action(action, state), "who wrote the play now")

    def test_no_op_is_always_valid(self):
        state = state_with("who wrote Hamlet?")
        action = parse_action(raw(action="no_op", source_query_id=None, new_query=""))
        self.assertEqual(validate_action(action, state), "original question?")


class ApplyTests(unittest.TestCase):
    def test_invalid_action_leaves_the_pool_identical_and_logs(self):
        state, ctx = state_with("original question?"), context()
        before = dict(state.query_pool)
        self.assertEqual(apply_opponent_action(ctx, state, raw(extra=1)),
                         "no_action")
        self.assertEqual(state.query_pool, before)
        row = state.decisions[-1]
        self.assertEqual((row["status"], row["parsed"]["action"]), ("invalid", "expand"))
        self.assertIn("unknown fields", row["error"])

    def test_zero_novelty_restores_the_pool(self):
        shared = [doc("same")]
        state = state_with("original question?")
        state.query_pool["original question?"] = shared
        state.evidence = None
        ctx = context({"zero_novelty_exit": True},
                      FakeRetriever({"a better query": [doc("same")]}))
        before = dict(state.query_pool)
        self.assertEqual(apply_opponent_action(ctx, state, raw()), EXIT_ZERO_NOVELTY)
        self.assertEqual(state.query_pool, before)

    def test_retriever_error_rolls_back(self):
        class RaisingRetriever(FakeRetriever):
            def search(self, query):
                raise RuntimeError("boom")

        state = state_with("original question?")
        before = dict(state.query_pool)
        outcome = apply_opponent_action(context(retriever=RaisingRetriever()), state, raw())
        self.assertEqual(outcome, "no_action")
        self.assertEqual(state.query_pool, before)
        self.assertIn("boom", state.decisions[-1]["error"])

    def test_optimize_uses_stable_id_not_levenshtein(self):
        state = state_with("who wrote it?", "who wrote the novel")
        before = dict(state.query_pool)
        self.assertEqual(find_most_similar_key(before, "who wrote it now"),
                         "who wrote it?")
        text = raw(action="optimize", source_query_id=2,
                   new_query="who wrote it now", reason="tighten")
        self.assertEqual(apply_opponent_action(context(), state, text), ACTION_APPLIED)
        self.assertEqual(set(state.query_pool), {"who wrote it?", "who wrote it now"})
        self.assertEqual(state.decisions[-1]["status"], "applied")
        self.assertEqual(register_query(state.queries, "who wrote it now", "x", 2), 3)


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, flags, outputs, docs=None):
        item = RecordingItem("original question?")
        pipeline = DragPlusPipeline(Flags(flags), ScriptedGenerator(list(outputs)),
                                    FakeRetriever(docs), IdentityTemplate(), "NQ")
        return item, pipeline.run([item])[0]

    def test_schema_prompt_lists_ids_and_changes_the_pool(self):
        text = raw(action="expand", source_query_id=1, new_query="second query now",
                   reason="missing detail")
        outputs = query_round("Opponent", text) + answer_rounds(1)
        flags = {"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
                 "opponent_action_schema": True}
        item, _ = self.run_pipeline(flags, outputs)
        system = item.output["QueryStage_Opponent Agent 0_Round0_InputPrompt"][0]["content"]
        self.assertIn("Q1: original question?", system)
        self.assertIn('"action": "optimize" | "expand" | "no_op"', system)
        self.assertEqual(set(item.output["QueryStage_QueryPool"]),
                         {"original question?", "second query now"})

    def test_default_opponent_prompt_is_unchanged(self):
        outputs = query_round("Proponent") + answer_rounds(1)
        item, _ = self.run_pipeline({"max_query_debate_rounds": 1,
                                     "max_answer_debate_rounds": 1}, outputs)
        system = item.output["QueryStage_Opponent Agent 0_Round0_InputPrompt"][0]["content"]
        self.assertNotIn("source_query_id", system)

    def test_judge_flag_changes_only_the_question_line(self):
        outputs = (query_round("Opponent", "weak. Query Expansion: second query now",
                               tag="r0")
                   + query_round("Proponent", tag="r1") + answer_rounds(1))
        base = {"max_query_debate_rounds": 2, "max_answer_debate_rounds": 1}
        off, _ = self.run_pipeline(base, outputs)
        on, _ = self.run_pipeline({**base, "judge_sees_refined_query": True}, outputs)
        key = "QueryStage_Moderator_Round1_InputPrompt"
        old = off.output[key][0]["content"].splitlines()
        new = on.output[key][0]["content"].splitlines()
        self.assertEqual(len(old), len(new))
        changed = [(left, right) for left, right in zip(old, new) if left != right]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0][0].startswith("Question: original question?"))
        self.assertEqual(changed[0][1], "Question: second query now")


if __name__ == "__main__":
    unittest.main()
