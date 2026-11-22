"""J12: log-prob Judge decisions, order balancing and the no-retry path."""
import copy
import unittest
from types import SimpleNamespace

from drag_plus.adapters import prompt_count
from drag_plus.flags import FlagError, Flags
from drag_plus.judge.logprob import (LABELS, ScoredGenerator, ScoredOutput, decide,
                                     margin, resolve_scored)
from drag_plus.pipeline import DragPlusPipeline
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator

RETRY_MARKER = "Your previous output was not a valid single-agent verdict"


class ScoredScriptedGenerator(ScriptedGenerator):
    """Text calls from the baseline queue, scored calls from a second queue."""

    def __init__(self, outputs, scored):
        super().__init__(outputs)
        self.scored = list(scored)
        self.score_calls = []

    def generate_with_scores(self, prompts, labels, **params):
        self.score_calls.append((copy.deepcopy(prompts), list(labels), dict(params)))
        if len(self.scored) < prompt_count(prompts):
            raise AssertionError("scripted scored outputs exhausted")
        return [ScoredOutput(text, scores)
                for text, scores in (self.scored.pop(0) for _ in range(prompt_count(prompts)))]


def scored(proponent, opponent, text="Proponent"):
    return (text, {"Proponent": proponent, "Opponent": opponent})


def build(scored_outputs, **flags):
    base = {"max_query_debate_rounds": 1, "response_rounds": 0,
            "judge_scoring": "logprob"}
    base.update(flags)
    generator = ScoredScriptedGenerator(
        ["proponent argument", "no action here", "The answer is: x"], scored_outputs)
    pipeline = DragPlusPipeline(Flags(base), generator, FakeRetriever(),
                                IdentityTemplate(), "NQ")
    item = RecordingItem("who wrote it?")
    return pipeline.run_item(item), generator, item


def system_content(prompt):
    return prompt[0]["content"]


def stable_lines(prompt):
    """Lines that must not move when only the two agents swap order."""
    return [line for line in prompt.splitlines()
            if not line.startswith(("1. ", "2. ", "Proponent Agent 0:", "Opponent Agent 0:"))]


def argument_order(prompt):
    lines = prompt.splitlines()
    start = lines.index("Agents Arguments:")
    return [line.split(":", 1)[0] for line in lines[start + 1:start + 3]]


class _Logprob:
    def __init__(self, value):
        self.logprob = value


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return list(self.vocab.get(text, []))


class FakeVLLMGenerator:
    """Return raw vLLM-shaped outputs: text plus per-position token log-probs."""

    def __init__(self, tokenizer, replies):
        self.tokenizer = tokenizer
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompts, **params):
        self.calls.append((copy.deepcopy(prompts), dict(params)))
        return [self._output(*self.replies.pop(0)) for _ in prompts]

    @staticmethod
    def _output(text, positions):
        maps = [{token: _Logprob(value) for token, value in position.items()}
                for position in positions]
        return SimpleNamespace(outputs=[SimpleNamespace(text=text, logprobs=maps)])


class ScoredGeneratorTests(unittest.TestCase):
    def test_first_token_scores_and_sampling_params(self):
        vocab = {"Proponent": [10, 11], " Proponent": [12],
                 "Opponent": [30, 31], " Opponent": [32]}
        generator = FakeVLLMGenerator(FakeTokenizer(vocab),
                                      [("Opponent", [{10: -2.0, 30: -1.0}])])
        adapter = ScoredGenerator(generator)
        outputs = adapter.generate_with_scores(["prompt"], labels=LABELS)
        self.assertEqual(outputs, [ScoredOutput("Opponent", {"Proponent": -2.0,
                                                             "Opponent": -1.0})])
        _, params = generator.calls[0]
        self.assertEqual((params["max_tokens"], params["logprobs"]), (2, 32))
        self.assertTrue(params["return_raw_output"])
        self.assertEqual(params["temperature"], 0.0)

    def test_colliding_first_tokens_sum_the_full_label(self):
        vocab = {"Proponent": [10, 11], " Opponent": [21], "Opponent": [10, 12]}
        positions = [{10: -1.0}, {11: -0.5, 12: -0.25}]
        generator = FakeVLLMGenerator(FakeTokenizer(vocab), [("x", positions)])
        scores = ScoredGenerator(generator).generate_with_scores(
            ["p"], labels=LABELS)[0].label_logprobs
        self.assertAlmostEqual(scores["Proponent"], -1.5)
        self.assertAlmostEqual(scores["Opponent"], -1.25)

    def test_absent_label_token_is_none_not_zero(self):
        vocab = {"Proponent": [10], "Opponent": [30]}
        generator = FakeVLLMGenerator(FakeTokenizer(vocab),
                                      [("Opponent", [{10: -2.0}])])
        scores = ScoredGenerator(generator).generate_with_scores(
            ["p"], labels=LABELS)[0].label_logprobs
        self.assertEqual((scores["Proponent"], scores["Opponent"]), (-2.0, None))


class DecisionTests(unittest.TestCase):
    def test_tau_thresholds(self):
        for tau in (-1.0, 0.0, 1.0):
            with self.subTest(tau=tau):
                self.assertEqual(decide(2.0, tau), "Opponent")
                self.assertEqual(decide(-2.0, tau), "Proponent")

    def test_tie_goes_to_proponent(self):
        verdict = resolve_scored([ScoredOutput("x", {"Proponent": -1.0,
                                                     "Opponent": -1.0})], 0.0, False)
        self.assertEqual((verdict.decision, verdict.score), ("Proponent", 0.0))

    def test_margin_is_opponent_minus_proponent(self):
        self.assertEqual(margin({"Proponent": -3.0, "Opponent": -1.0}), 2.0)
        self.assertIsNone(margin({"Proponent": -3.0}))
        self.assertIsNone(margin({"Proponent": float("inf"), "Opponent": -1.0}))

    def test_order_balance_averages_both_orders(self):
        po, op = scored(-3.0, -1.0), scored(-1.0, -2.0)
        verdict = resolve_scored([ScoredOutput(*po), ScoredOutput(*op)], 0.0, True)
        self.assertEqual((verdict.decision, verdict.score), ("Opponent", 0.5))
        self.assertEqual(resolve_scored(
            [ScoredOutput(*po), ScoredOutput(*op)], 1.0, True).decision, "Proponent")

    def test_order_balance_needs_both_orders(self):
        po, op = scored(-3.0, -1.0, text="Opponent"), scored(None, -2.0)
        verdict = resolve_scored([ScoredOutput(*po), ScoredOutput(*op)], 0.0, True)
        self.assertEqual((verdict.decision, verdict.status, verdict.score),
                         ("Opponent", "fallback_generative", None))


class FlagTests(unittest.TestCase):
    def test_defaults_keep_the_generative_path(self):
        flags = Flags()
        self.assertEqual((flags.judge_scoring, flags.judge_order_balance, flags.judge_tau),
                         ("generative", False, 0.0))

    def test_logprob_flags_validate(self):
        flags = Flags({"judge_scoring": "logprob", "judge_order_balance": True,
                       "judge_tau": 1})
        self.assertEqual(flags.judge_tau, 1.0)
        with self.assertRaises(FlagError):
            Flags({"judge_order_balance": True})
        with self.assertRaises(FlagError):
            Flags({"judge_scoring": "probability"})


class IntegrationTests(unittest.TestCase):
    def test_decision_and_score_are_recorded_without_retry(self):
        state, generator, item = build([scored(-3.0, -1.0, text="Proponent")])
        self.assertEqual(state.decisions,
                         [{"stage": "query", "round": 0, "decision": "Opponent",
                           "status": "valid_logprob", "score": 2.0}])
        self.assertEqual(state.judge_score, 2.0)
        self.assertIn(("QueryStage_Moderator_Round0_Score", 2.0), item.calls)
        self.assertEqual(len(generator.score_calls), 1)
        self.assertFalse(any(RETRY_MARKER in str(prompt)
                             for prompt, _ in generator.calls))
        self.assertEqual([params for _, params in generator.calls], [{}, {}, {}])

    def test_order_balance_is_one_two_prompt_call(self):
        state, generator, _ = build(
            [scored(-3.0, -1.0), scored(-2.0, -1.0)], judge_order_balance=True)
        self.assertEqual(len(generator.score_calls), 1)
        prompts, labels, _ = generator.score_calls[0]
        self.assertEqual(labels, list(LABELS))
        self.assertEqual(len(prompts), 2)
        po, op = (system_content(prompt) for prompt in prompts)
        self.assertEqual(stable_lines(po), stable_lines(op))
        self.assertEqual(argument_order(po),
                         ["Proponent Agent 0", "Opponent Agent 0"])
        self.assertEqual(argument_order(op),
                         ["Opponent Agent 0", "Proponent Agent 0"])
        self.assertIn("1. Proponent Agent:", po)
        self.assertIn("1. Opponent Agent:", op)
        self.assertEqual((state.decisions[0]["decision"], state.judge_score),
                         ("Opponent", 1.5))

    def test_missing_label_logs_fallback_without_a_second_call(self):
        state, generator, item = build([scored(-1.0, None, text="Opponent")])
        self.assertEqual(state.decisions[-1]["status"], "fallback_generative")
        self.assertEqual(state.decisions[-1]["decision"], "Opponent")
        self.assertIn("judge_fallback", [entry["decision"] for entry in state.decisions])
        self.assertIn(("QueryStage_Moderator_Round0_Fallback", "fallback_generative"),
                      item.calls)
        self.assertEqual(len(generator.score_calls), 1)
        self.assertEqual(len(generator.calls), 3)


if __name__ == "__main__":
    unittest.main()
