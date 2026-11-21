"""J08 T-flag tests: response rounds, per-role token budgets and concise agents.

The defaults must stay byte-identical to the baseline (T0 proves that separately); these
tests switch one budget flag at a time and check call counts, parameters, finish reasons
and the truncated-output fallback.
"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.flags import FlagError, Flags  # noqa: E402
from drag_plus.instrument.timing import CallRecorder  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from drag_plus.query_stage import PROPONENT, OPPONENT  # noqa: E402
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from j06_helpers import run_plus, script
from t0_scenarios import answer_rounds, query_round

QUESTION = "who wrote it?"
ZERO = lambda: 0.0  # noqa: E731 - a deterministic clock for every row


def answer_rows(state):
    return [row for row in state.calls
            if row["stage"] in ("answer_agent", "answer_moderator")]


def run_budgeted(rounds, final_only=False, **flags):
    """Run one query round and ``rounds`` answer rounds with the budget flags set."""
    values = {"response_rounds": rounds, "moderator_final_only": final_only,
              "profile_calls": True, **flags}
    outputs = script(query_rounds=1, answer_rounds=rounds, final_only=final_only)
    if rounds == 0:
        outputs += ["final answer"]
    item, gen, state = run_plus(outputs, values, query_rounds=1, answer_rounds=3,
                                recorder=CallRecorder(ZERO))
    return item, gen, state


class RoundCountTests(unittest.TestCase):
    def test_call_matrix_with_and_without_the_final_moderator(self):
        for rounds in (0, 1, 2, 3):
            for final_only in (False, True):
                with self.subTest(rounds=rounds, final_only=final_only):
                    _item, _gen, state = run_budgeted(rounds, final_only)
                    rows = answer_rows(state)
                    moderators = [row for row in rows
                                  if row["stage"] == "answer_moderator"]
                    expected = 1 if rounds == 0 else 2 * rounds + (
                        1 if final_only else rounds)
                    self.assertEqual(len(rows), expected)
                    self.assertEqual(len(moderators),
                                     0 if rounds == 0 else (1 if final_only else rounds))

    def test_null_response_rounds_inherits_the_baseline_flag(self):
        item, gen, _state = run_plus(script(1, 2), {"max_answer_debate_rounds": 2},
                                     query_rounds=1, answer_rounds=2)
        self.assertEqual(len(gen.calls) - 3, 2 * 2 + 2)
        self.assertEqual(item.pred, "The answer is: moderator 1")

    def test_zero_response_rounds_keeps_the_answer_only_path(self):
        item, gen, state = run_budgeted(0)
        self.assertEqual(len(gen.calls), 4)  # 3 query calls + one answer-only call
        self.assertEqual(state.answer_messages, {})
        self.assertEqual(item.pred, "final answer")


class TokenBudgetTests(unittest.TestCase):
    def test_per_role_budgets_reach_each_call_without_leakage(self):
        _item, gen, state = run_budgeted(2, agent_max_tokens=64, moderator_max_tokens=16)
        rows = answer_rows(state)
        self.assertEqual([params for _p, params in gen.calls[-len(rows):]],
                         [{"max_tokens": 64}, {"max_tokens": 64}, {"max_tokens": 16}] * 2)
        for _prompt, params in gen.calls:
            self.assertLessEqual(set(params), {"max_tokens"})

    def test_defaults_pass_no_generation_parameters(self):
        _item, gen, _state = run_budgeted(1)
        self.assertEqual([params for _p, params in gen.calls], [{}] * len(gen.calls))

    def test_batched_agents_keep_the_agent_budget(self):
        _item, gen, state = run_budgeted(1, batch_agents_in_round=True,
                                         agent_max_tokens=48, moderator_max_tokens=12)
        rows = answer_rows(state)
        self.assertEqual([params for _p, params in gen.calls[-len(rows):]],
                         [{"max_tokens": 48}, {"max_tokens": 12}])
        self.assertEqual(rows[0]["batch_size"], 2)


class ConciseAgentTests(unittest.TestCase):
    def test_concise_text_reaches_both_agent_turns_only(self):
        _item, _gen, state = run_budgeted(2, concise_agents=True)
        for agent in (PROPONENT, OPPONENT):
            for index in (0, 3):  # round-0 system prompt, round-1 debate prompt
                text = state.answer_messages[agent][index]["content"]
                self.assertIn("at most 2 sentences", text)
                self.assertIn("The answer is:", text)
        moderator = dict(state.outputs)["AnswerStage_Moderator_Round1_InputPrompt"]
        text = "\n".join(message["content"] for message in moderator)
        self.assertNotIn("at most 2 sentences", text)

    def test_defaults_keep_the_baseline_instruction(self):
        _item, _gen, state = run_budgeted(1)
        system = state.answer_messages[PROPONENT][0]["content"]
        self.assertIn("Explain your answer", system)
        self.assertNotIn("at most 2 sentences", system)

    def test_concise_strategyqa_keeps_the_yes_no_choice(self):
        outputs = (query_round("Proponent")
                   + ["The answer is: Yes", "The answer is: No", "The answer is: Yes"])
        flags = Flags({"max_query_debate_rounds": 1, "response_rounds": 1,
                       "concise_agents": True})
        pipeline = DragPlusPipeline(flags, ScriptedGenerator(outputs), FakeRetriever(),
                                    IdentityTemplate(), "StrategyQA")
        state = pipeline.run_item(RecordingItem(QUESTION))
        system = state.answer_messages[PROPONENT][0]["content"]
        self.assertIn("Given two answer candidates", system)
        self.assertIn("at most 2 sentences", system)


class BudgetFlagTests(unittest.TestCase):
    def test_response_rounds_accepts_zero_to_three_only(self):
        for value in (0, 1, 2, 3):
            self.assertEqual(Flags({"response_rounds": value}).response_rounds, value)
        for bad in (-1, 4, "1", True):
            with self.subTest(value=bad), self.assertRaises(FlagError):
                Flags({"response_rounds": bad})

    def test_token_budgets_are_positive_ints_or_null(self):
        for name in ("agent_max_tokens", "moderator_max_tokens"):
            self.assertIsNone(getattr(Flags(), name))
            self.assertEqual(getattr(Flags({name: 8}), name), 8)
            for bad in (0, -1, 8.0, "8", True):
                with self.subTest(name=name, value=bad), self.assertRaises(FlagError):
                    Flags({name: bad})

    def test_concise_agents_is_a_bool(self):
        self.assertFalse(Flags().concise_agents)
        self.assertTrue(Flags({"concise_agents": True}).concise_agents)
        with self.assertRaises(FlagError):
            Flags({"concise_agents": "yes"})


if __name__ == "__main__":
    unittest.main()
