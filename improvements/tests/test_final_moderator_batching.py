"""J06 T-flag test, step 1: ``moderator_final_only``.

The flag must be answer-preserving: with it on, the answer stage makes exactly 2R + 1
generations, and the final moderator prompt, every agent prompt and ``pred`` equal the
frozen v1.0.1 pipeline's. Only the rounds 0..R-2 moderator outputs are absent.
"""
import unittest

import frozen
from drag_plus.flags import Flags
from drag_plus.instrument.timing import CallRecorder
from drag_plus.pipeline import DragPlusPipeline
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from t0_scenarios import query_round

QUESTION = "who wrote it?"
MODERATOR_MARKER = "You are a moderator in a debate competition"


class FrozenBaseline(frozen.DebateAugmentedRAG):
    def evaluate(self, dataset, do_eval=True):
        return dataset


def script(answer_rounds, final_only=False):
    """One query round that ends, then R answer rounds.

    A final-only script omits the moderator outputs of rounds 0..R-2: those calls are
    never made, so the baseline's script would leave them unconsumed.
    """
    outputs = query_round("Proponent")
    for r in range(answer_rounds):
        outputs += [f"proponent answer {r}", f"opponent answer {r}"]
        if not final_only or r == answer_rounds - 1:
            outputs.append(f"The answer is: moderator {r}")
    return outputs


def run_baseline(outputs, rounds):
    gen, ret, template = ScriptedGenerator(outputs), FakeRetriever(), IdentityTemplate()
    pipeline = FrozenBaseline(
        {"dataset_name": "NQ"}, prompt_template=template, max_query_debate_rounds=1,
        max_answer_debate_rounds=rounds, generator=gen, retriever=ret)
    item = RecordingItem(QUESTION)
    pipeline.run([item])
    return item, gen


def run_plus(outputs, values=None, rounds=2, recorder=None):
    gen, ret, template = ScriptedGenerator(outputs), FakeRetriever(), IdentityTemplate()
    flags = {"max_query_debate_rounds": 1, "max_answer_debate_rounds": rounds,
             **(values or {})}
    pipeline = DragPlusPipeline(Flags(flags), gen, ret, template, "NQ", recorder=recorder)
    item = RecordingItem(QUESTION)
    return item, gen, pipeline.run_item(item)


def prompts_of(calls):
    return [prompts for prompts, _params in calls]


def is_moderator_prompt(prompt):
    return isinstance(prompt, list) and any(
        MODERATOR_MARKER in message.get("content", "") for message in prompt)


def drop_early_moderator_prompts(prompts, rounds):
    """Baseline prompts minus the moderator prompts of rounds 0..R-2, in order."""
    kept, seen = [], 0
    for prompt in prompts:
        if is_moderator_prompt(prompt):
            seen += 1
            if seen < rounds:
                continue
        kept.append(prompt)
    return kept


def is_early_moderator(key, rounds):
    if not key.startswith("AnswerStage_Moderator_Round"):
        return False
    return int(key.split("Round")[1].split("_")[0]) < rounds - 1


def drop_early_moderator_outputs(calls, rounds):
    """The baseline transcript minus the early moderator outputs and their pred writes."""
    total_preds = sum(1 for key, _ in calls if key == "pred")
    kept, seen = [], 0
    for key, value in calls:
        if key == "pred" and seen < total_preds - 1:
            seen += 1
            continue
        if not is_early_moderator(key, rounds):
            kept.append((key, value))
    return kept


class FinalModeratorTests(unittest.TestCase):
    def answer_generations(self, rounds):
        _item, gen, state = run_plus(
            script(rounds, final_only=True),
            {"moderator_final_only": True, "profile_calls": True}, rounds=rounds,
            recorder=CallRecorder(lambda: 0.0))
        rows = [row for row in state.calls
                if row["stage"] in ("answer_agent", "answer_moderator")]
        return rows, gen

    def test_answer_stage_makes_two_r_plus_one_generations(self):
        for rounds in (1, 2, 3):
            with self.subTest(rounds=rounds):
                rows, gen = self.answer_generations(rounds)
                self.assertEqual(sum(row["batch_size"] for row in rows), 2 * rounds + 1)
                self.assertEqual(len([r for r in rows if r["stage"] == "answer_moderator"]), 1)
                self.assertEqual(len(gen.calls), 3 + 2 * rounds + 1)

    def test_final_prompt_and_pred_equal_the_baseline(self):
        for rounds in (1, 2, 3):
            with self.subTest(rounds=rounds):
                base, base_gen = run_baseline(script(rounds), rounds)
                item, plus_gen, _ = run_plus(script(rounds, final_only=True),
                                             {"moderator_final_only": True}, rounds=rounds)
                last = f"AnswerStage_Moderator_Round{rounds - 1}_InputPrompt"
                self.assertEqual(item.output[last], base.output[last])
                self.assertEqual(item.pred, base.pred)
                self.assertEqual(prompts_of(plus_gen.calls),
                                 drop_early_moderator_prompts(prompts_of(base_gen.calls),
                                                              rounds))

    def test_only_the_last_moderator_keys_remain(self):
        base, _ = run_baseline(script(2), 2)
        item, _, _ = run_plus(script(2, final_only=True), {"moderator_final_only": True})
        self.assertEqual(item.calls, drop_early_moderator_outputs(base.calls, 2))
        self.assertEqual(item.pred, base.pred)

    def test_answer_only_path_is_unchanged(self):
        outputs = query_round("Proponent") + ["final answer"]
        base, base_gen = run_baseline(outputs, 0)
        for final_only in (False, True):
            with self.subTest(final_only=final_only):
                item, gen, _ = run_plus(outputs, {"moderator_final_only": final_only},
                                        rounds=0)
                self.assertEqual(prompts_of(gen.calls), prompts_of(base_gen.calls))
                self.assertEqual(item.calls, base.calls)
                self.assertEqual(item.pred, base.pred)

    def test_flag_off_makes_the_baseline_call_count(self):
        base, base_gen = run_baseline(script(3), 3)
        _item, gen, _ = run_plus(script(3), rounds=3)
        self.assertEqual(len(gen.calls), len(base_gen.calls))
        self.assertEqual(len(gen.calls), 3 + 3 * 3)


if __name__ == "__main__":
    unittest.main()
