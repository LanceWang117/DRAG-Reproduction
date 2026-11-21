"""J06 T-flag tests: final-only moderator and intra-round agent batching.

Both flags are answer-preserving. The reference is the frozen v1.0.1 pipeline: with a
flag on, every emitted output and every generator prompt must match it, except for the
two documented groupings (one 2-prompt call per round; early moderator keys absent).
"""
import unittest

from drag_plus.adapters import GeneratorContractError, prompt_count
from drag_plus.instrument.timing import CallRecorder
from drag_plus.query_stage import AGENTS
from j06_helpers import (drop_early_moderator_outputs, drop_early_moderator_prompts,
                         prompts_of, run_baseline, run_plus, script)

PROPONENT, OPPONENT = AGENTS


class FinalModeratorTests(unittest.TestCase):
    def answer_generations(self, rounds):
        _item, gen, state = run_plus(
            script(1, rounds, final_only=True),
            {"moderator_final_only": True, "profile_calls": True},
            query_rounds=1, answer_rounds=rounds, recorder=CallRecorder(lambda: 0.0))
        rows = [r for r in state.calls if r["stage"] in ("answer_agent", "answer_moderator")]
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
                base, base_gen = run_baseline(script(1, rounds), query_rounds=1,
                                              answer_rounds=rounds)
                item, plus_gen, _ = run_plus(
                    script(1, rounds, final_only=True), {"moderator_final_only": True},
                    query_rounds=1, answer_rounds=rounds)
                last = f"AnswerStage_Moderator_Round{rounds - 1}_InputPrompt"
                self.assertEqual(item.output[last], base.output[last])
                self.assertEqual(item.pred, base.pred)
                self.assertEqual(prompts_of(plus_gen.calls),
                                 drop_early_moderator_prompts(prompts_of(base_gen.calls),
                                                              rounds))

    def test_only_the_last_moderator_keys_remain(self):
        base, _ = run_baseline(script(2, 2))
        item, _, _ = run_plus(script(2, 2, final_only=True), {"moderator_final_only": True})
        self.assertEqual(item.calls, drop_early_moderator_outputs(base.calls, 2))
        self.assertEqual(item.pred, base.pred)

    def test_answer_only_path_is_unchanged(self):
        outputs = script(1, 0) + ["final answer"]
        base, base_gen = run_baseline(outputs, query_rounds=1, answer_rounds=0)
        for final_only in (False, True):
            with self.subTest(final_only=final_only):
                item, gen, _ = run_plus(outputs, {"moderator_final_only": final_only},
                                        query_rounds=1, answer_rounds=0)
                self.assertEqual(prompts_of(gen.calls), prompts_of(base_gen.calls))
                self.assertEqual(item.calls, base.calls)
                self.assertEqual(item.pred, "final answer")

    def test_flag_off_makes_the_baseline_call_count(self):
        base, base_gen = run_baseline(script(2, 3), answer_rounds=3)
        _item, gen, _ = run_plus(script(2, 3), answer_rounds=3)
        self.assertEqual(len(gen.calls), len(base_gen.calls))
        self.assertEqual(len(gen.calls), 2 * 3 + 3 * 3)


class BatchingTests(unittest.TestCase):
    def test_one_two_prompt_call_per_round_in_both_stages(self):
        _item, gen, _ = run_plus(script(), {"batch_agents_in_round": True})
        sizes = [prompt_count(prompts) for prompts, _ in gen.calls]
        self.assertEqual(sizes, [2, 1, 2, 1, 2, 1, 2, 1])

    def test_each_prompt_is_the_baseline_prompt_in_order(self):
        base, base_gen = run_baseline(script())
        item, plus_gen, _ = run_plus(script(), {"batch_agents_in_round": True})
        self.assertEqual(prompts_of(plus_gen.calls), prompts_of(base_gen.calls))
        self.assertEqual(item.calls, base.calls)
        self.assertEqual(item.pred, base.pred)

    def test_outputs_are_mapped_to_roles_by_position(self):
        item, gen, state = run_plus(script(), {"batch_agents_in_round": True})
        batches = [prompts for prompts, _ in gen.calls if prompt_count(prompts) == 2]
        query_batches = [batch for batch in batches
                         if "Argue that the current retrieved content" in batch[0][0]["content"]]
        first, last = query_batches[0], query_batches[-1]
        self.assertIn("challenge the sufficiency of the current retrieved content",
                      first[1][0]["content"])
        self.assertEqual(state.query_messages[PROPONENT], [last[0], "proponent argument "])
        self.assertEqual(state.query_messages[OPPONENT], [last[1], "no action here"])
        self.assertEqual(item.output[f"AnswerStage_{PROPONENT}_Round0_Output"],
                         "proponent answer 0")
        self.assertEqual(item.output[f"AnswerStage_{OPPONENT}_Round0_Output"],
                         "opponent answer 0")

    def test_a_batched_call_is_one_timing_row_with_batch_size_two(self):
        _item, _gen, state = run_plus(
            script(1, 1), {"batch_agents_in_round": True, "profile_calls": True},
            query_rounds=1, answer_rounds=1, recorder=CallRecorder(lambda: 0.0))
        rows = [(row["stage"], row["role"], row["batch_size"]) for row in state.calls
                if row["stage"] in ("query_agent", "answer_agent")]
        self.assertEqual(rows, [("query_agent", "agents", 2),
                                ("answer_agent", "agents", 2)])

    def test_flag_off_sends_one_prompt_at_a_time(self):
        _item, gen, _ = run_plus(script())
        self.assertEqual({prompt_count(prompts) for prompts, _ in gen.calls}, {1})


class CombinationTests(unittest.TestCase):
    def test_all_four_combinations_preserve_prompts_and_pred(self):
        base, base_gen = run_baseline(script())
        base_prompts = prompts_of(base_gen.calls)
        for final_only in (False, True):
            for batched in (False, True):
                with self.subTest(final_only=final_only, batched=batched):
                    item, gen, _ = run_plus(
                        script(final_only=final_only),
                        {"moderator_final_only": final_only,
                         "batch_agents_in_round": batched})
                    expected = (drop_early_moderator_prompts(base_prompts, 2) if final_only
                                else base_prompts)
                    self.assertEqual(prompts_of(gen.calls), expected)
                    self.assertEqual(item.pred, base.pred)
                    sizes = [prompt_count(prompts) for prompts, _ in gen.calls]
                    self.assertEqual(set(sizes), {2, 1} if batched else {1})
                    moderators = [k for k, _ in item.calls
                                  if k.startswith("AnswerStage_Moderator_Round")]
                    self.assertEqual(len(moderators), 2 if final_only else 4)


class ContractTests(unittest.TestCase):
    class ShortGenerator:
        def generate(self, prompts, **params):
            return ["only one output"]

    class SwappedGenerator:
        """Correct outputs in the wrong order, so a positional mapping must show it."""

        def generate(self, prompts, **params):
            count = prompt_count(prompts)
            return [f"output {i}" for i in reversed(range(count))]

    def test_a_short_batch_is_a_clear_error(self):
        with self.assertRaisesRegex(GeneratorContractError, "1 outputs for 2 prompts"):
            run_plus(script(), {"batch_agents_in_round": True},
                     generator=self.ShortGenerator())

    def test_roles_follow_output_position_not_output_text(self):
        _item, _gen, state = run_plus(script(), {"batch_agents_in_round": True},
                                     generator=self.SwappedGenerator())
        self.assertEqual(state.query_messages[PROPONENT][1], "output 1")
        self.assertEqual(state.query_messages[OPPONENT][1], "output 0")


if __name__ == "__main__":
    unittest.main()
