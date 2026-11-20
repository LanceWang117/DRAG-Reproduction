"""Scripted scenarios for T0: outputs in the exact order the pipeline asks for them."""
from dataclasses import dataclass
from typing import Tuple

QUESTION = "who wrote it?"


@dataclass(frozen=True)
class Scenario:
    name: str
    outputs: Tuple[str, ...]
    query_rounds: int = 3
    answer_rounds: int = 3
    dataset: str = "NQ"
    questions: Tuple[str, ...] = (QUESTION,)
    expect_pool_keys: Tuple[str, ...] = ()


def query_round(judge, opponent="no action here", retry=None, tag=""):
    outputs = [f"proponent argument {tag}", opponent, judge]
    return outputs + ([retry] if retry is not None else [])


def answer_rounds(count, tag=""):
    outputs = []
    for r in range(count):
        outputs += [f"proponent answer {tag}{r}", f"opponent answer {tag}{r}",
                    f"The answer is: moderator {tag}{r}"]
    return outputs


EXPANSION = "weak. Query Expansion: second query"


def _query_stage_scenarios():
    optimization = "weak. Query Optimization: who wrote it -> who wrote the novel"
    bare = "weak. Query Optimization: second quer"
    return [
        Scenario("a_valid_proponent_round0", tuple(query_round("Proponent") + answer_rounds(3))),
        Scenario("b_expansion_then_proponent", tuple(
            query_round("Opponent", EXPANSION) + query_round("Proponent") + answer_rounds(1)),
            answer_rounds=1, expect_pool_keys=("second query",)),
        Scenario("c_optimization_replaces_original", tuple(
            query_round("Opponent", optimization) + query_round("Proponent") + answer_rounds(1)),
            answer_rounds=1, expect_pool_keys=("who wrote the novel",)),
        Scenario("d_ambiguous_then_retry", tuple(
            query_round("Proponent Agent 1:\nOpponent Agent 1:", EXPANSION, "Opponent")
            + query_round("Proponent") + answer_rounds(1)),
            answer_rounds=1, expect_pool_keys=("second query",)),
        Scenario("e_unresolved_twice", tuple(
            query_round("unclear", retry="still unclear", tag="r0")
            + query_round("unclear", retry="still unclear", tag="r1") + answer_rounds(1)),
            query_rounds=2, answer_rounds=1),
        Scenario("g_opponent_without_action", tuple(
            query_round("Opponent", "needs more but names no action") + answer_rounds(1)),
            answer_rounds=1),
        Scenario("j_bare_optimization_picks_closest_key", tuple(
            query_round("Opponent", EXPANSION, tag="r0")
            + query_round("Opponent", bare, tag="r1")
            + query_round("Proponent", tag="r2") + answer_rounds(1)),
            answer_rounds=1, expect_pool_keys=("second quer", QUESTION)),
    ]


def _pool_edge_scenarios():
    tie = "weak. Query Optimization: who wrote it. -> new query"
    onto_existing = "weak. Query Optimization: second query -> who wrote it?"
    return [
        Scenario("m_equal_distance_keeps_first_key", tuple(
            query_round("Opponent", "weak. Query Expansion: who wrote it!", tag="r0")
            + query_round("Opponent", tie, tag="r1")
            + query_round("Proponent", tag="r2") + answer_rounds(1)),
            answer_rounds=1, expect_pool_keys=("who wrote it!", "new query")),
        Scenario("n_optimization_onto_existing_key", tuple(
            query_round("Opponent", EXPANSION, tag="r0")
            + query_round("Opponent", onto_existing, tag="r1")
            + query_round("Proponent", tag="r2") + answer_rounds(1)),
            answer_rounds=1, expect_pool_keys=(QUESTION,)),
        Scenario("o_padded_question_is_stripped_for_the_pool_key", tuple(
            query_round("Proponent") + answer_rounds(1)),
            answer_rounds=1, questions=("  padded question? \n",),
            expect_pool_keys=("padded question?",)),
    ]


def _round_and_dataset_scenarios():
    return [
        Scenario("f_strategyqa_prompts", tuple(query_round("Proponent") + answer_rounds(2)),
                 query_rounds=1, answer_rounds=2, dataset="StrategyQA"),
        Scenario("h_zero_query_rounds", tuple(answer_rounds(2)),
                 query_rounds=0, answer_rounds=2),
        Scenario("i_zero_answer_rounds", tuple(query_round("Proponent") + ["final answer"]),
                 query_rounds=1, answer_rounds=0),
        Scenario("k_strategyqa_answer_only", tuple(query_round("Proponent") + ["Yes"]),
                 query_rounds=1, answer_rounds=0, dataset="StrategyQA"),
        Scenario("l_two_questions_do_not_leak", tuple(
            query_round("Opponent", EXPANSION, tag="q0") + query_round("Proponent", tag="q0b")
            + answer_rounds(2, "q0-") + query_round("Proponent", tag="q1")
            + answer_rounds(2, "q1-")),
            query_rounds=2, answer_rounds=2, questions=(QUESTION, "second question?")),
    ]


SCENARIOS = tuple(_query_stage_scenarios() + _pool_edge_scenarios()
                  + _round_and_dataset_scenarios())
