"""Query-stage debate: retrieve, let two agents argue, let the Judge decide, refine.

Reproduces the v1.0.1 behaviour including its quirks: the Judge runs every round (even
when its verdict cannot change anything), an unresolved verdict skips to the next round,
Opponent text without an action ends the stage, and optimisation deletes the pool entry
closest to the quoted original by edit distance.
"""
from __future__ import annotations

from typing import Optional

import Levenshtein

from . import prompts
from .adapters import BATCH_ROLE, StageContext
from .judge_decision import JudgeDecision, resolve_judge_decision
from .query_norm import is_duplicate_query
from .state import QuestionState

PROPONENT = "Proponent Agent 0"
OPPONENT = "Opponent Agent 0"
AGENTS = (PROPONENT, OPPONENT)

ACTION_APPLIED = "applied"
ACTION_NONE = "no_action"
EXIT_DUPLICATE = "duplicate"
EXIT_ZERO_NOVELTY = "zero_novelty"


class QueryPoolError(ValueError):
    """The Opponent's proposal could not be applied to the query pool."""


def find_most_similar_key(pool: dict, target: str) -> Optional[str]:
    best_key, best = None, float("inf")
    for key in pool:
        distance = Levenshtein.distance(key, target)
        if distance < best:
            best, best_key = distance, key
    return best_key


def _split_optimization(instruction: str):
    if "->" in instruction:
        parts = instruction.split("->")
        return parts[0].strip(), parts[1].strip()
    return instruction, instruction


def _pool_ids(pool: dict) -> set:
    return {passage.get("id") for passages in pool.values() for passage in passages
            if passage.get("id") is not None}


def _has_new_id(passages: list, before: set) -> bool:
    found = {passage.get("id") for passage in passages if passage.get("id") is not None}
    return bool(found - before)


def _is_pool_duplicate(query: str, pool: dict) -> bool:
    return is_duplicate_query(query, pool)


def _optimize(ctx: StageContext, state: QuestionState, instruction: str) -> str:
    pool = state.query_pool
    original, new_query = _split_optimization(instruction)
    old_key = find_most_similar_key(pool, original)
    if ctx.flags.skip_duplicate_query and _is_pool_duplicate(new_query, pool):
        return EXIT_DUPLICATE
    before = _pool_ids(pool)
    saved = list(pool.items()) if ctx.flags.zero_novelty_exit else None
    pool.pop(old_key)
    passages = ctx.backends.search(new_query, call=state.call("retrieval", "retriever"))
    if ctx.flags.zero_novelty_exit and not _has_new_id(passages, before):
        pool.clear()
        pool.update(saved)
        return EXIT_ZERO_NOVELTY
    pool[new_query] = passages
    return ACTION_APPLIED


def _expand(ctx: StageContext, state: QuestionState, new_query: str) -> str:
    pool = state.query_pool
    if ctx.flags.skip_duplicate_query and _is_pool_duplicate(new_query, pool):
        return EXIT_DUPLICATE
    before = _pool_ids(pool)
    passages = ctx.backends.search(new_query, call=state.call("retrieval", "retriever"))
    if ctx.flags.zero_novelty_exit and not _has_new_id(passages, before):
        return EXIT_ZERO_NOVELTY
    pool[new_query] = passages
    return ACTION_APPLIED


def _apply_action(ctx: StageContext, state: QuestionState, text: str) -> str:
    if "Query Optimization:" in text:
        return _optimize(ctx, state, text.split("Query Optimization:")[1].strip())
    if "Query Expansion:" in text:
        return _expand(ctx, state, text.split("Query Expansion:")[1].strip())
    return ACTION_NONE


def maintain_query_pool(ctx: StageContext, state: QuestionState,
                        opponent_output: str) -> str:
    """Apply the Opponent's action in place; returns an action or exit outcome.

    ``EXIT_DUPLICATE`` means the duplicate guard skipped the search and the stage must
    end; ``EXIT_ZERO_NOVELTY`` means the search added no passage ID and the pool is
    restored to its pre-action contents. ``ACTION_NONE`` means no action was proposed.
    """
    try:
        return _apply_action(ctx, state, opponent_output)
    except Exception as error:
        raise QueryPoolError(
            f"Error in maintaining the query pool for Opponent output {opponent_output!r}"
        ) from error


def _agent_prompt(ctx: StageContext, state: QuestionState, agent: str):
    messages = [prompts.query_agent_system(agent),
                prompts.query_round_user(state.question, state.query_pool)]
    return ctx.backends.render(messages)


def _record_turn(state: QuestionState, round_index: int, agent: str,
                 prompt, output: str) -> None:
    state.emit(f"QueryStage_{agent}_Round{round_index}_InputPrompt", prompt)
    state.emit(f"QueryStage_{agent}_Round{round_index}_Output", output)
    state.query_messages[agent] = [prompt, output]


def _agent_turn(ctx: StageContext, state: QuestionState, round_index: int, agent: str) -> None:
    prompt = _agent_prompt(ctx, state, agent)
    output = ctx.backends.complete(prompt, call=state.call("query_agent", agent))
    _record_turn(state, round_index, agent, prompt, output)


def _agent_round(ctx: StageContext, state: QuestionState, round_index: int) -> None:
    """Both agents of one round in a single 2-prompt call; outputs map by position."""
    prompts = [_agent_prompt(ctx, state, agent) for agent in AGENTS]
    outputs = ctx.backends.complete_batch(prompts, call=state.call("query_agent", BATCH_ROLE))
    for agent, prompt, output in zip(AGENTS, prompts, outputs):
        _record_turn(state, round_index, agent, prompt, output)


def _run_agents(ctx: StageContext, state: QuestionState, round_index: int) -> None:
    if ctx.flags.batch_agents_in_round:
        _agent_round(ctx, state, round_index)
        return
    for agent in AGENTS:
        _agent_turn(ctx, state, round_index, agent)


def _judge(ctx: StageContext, state: QuestionState, round_index: int) -> JudgeDecision:
    prefix = f"QueryStage_Moderator_Round{round_index}"
    outputs = {agent: state.query_messages[agent][1] for agent in state.query_messages}
    message = [prompts.query_moderator_system(outputs, state.question, state.query_pool)]
    prompt = ctx.backends.render(message)
    output = ctx.backends.complete(prompt, call=state.call("query_judge", "judge"))
    state.emit(f"{prefix}_InputPrompt", prompt)
    state.emit(f"{prefix}_Output", output)

    def retry() -> str:
        retry_message = message + [{"role": "assistant", "content": output},
                                   prompts.judge_retry_user()]
        retry_prompt = ctx.backends.render(retry_message)
        retry_output = ctx.backends.complete(
            retry_prompt, call=state.call("query_judge_retry", "judge"))
        state.emit(f"{prefix}_RetryInputPrompt", retry_prompt)
        state.emit(f"{prefix}_RetryOutput", retry_output)
        return retry_output

    verdict = resolve_judge_decision(output, retry)
    state.emit(f"{prefix}_Decision", verdict.decision)
    state.emit(f"{prefix}_DecisionStatus", verdict.status)
    state.decisions.append({"stage": "query", "round": round_index,
                            "decision": verdict.decision, "status": verdict.status})
    return verdict


def _continues_after(ctx: StageContext, state: QuestionState, verdict: JudgeDecision) -> bool:
    """True when the debate goes on to another round after this verdict."""
    if verdict.decision is None:
        return True
    if verdict.decision == "Proponent":
        return False
    outcome = maintain_query_pool(ctx, state, state.query_messages[OPPONENT][1])
    if outcome in (EXIT_DUPLICATE, EXIT_ZERO_NOVELTY):
        state.decisions.append({"stage": "query", "round": state.round,
                                "decision": "query_exit", "status": outcome})
        return False
    return outcome == ACTION_APPLIED


def run_query_stage(ctx: StageContext, state: QuestionState) -> None:
    """Fill ``state.query_pool``; the stage ends when the Judge is satisfied."""
    state.stage = "query"
    rounds = ctx.flags.max_query_debate_rounds
    for round_index in range(max(rounds, 1)):
        state.round = round_index
        if round_index == 0:
            question = state.question
            state.query_pool[question.strip()] = ctx.backends.search(
                question, call=state.call("retrieval", "retriever"))
        if rounds == 0:
            return
        _run_agents(ctx, state, round_index)
        if not _continues_after(ctx, state, _judge(ctx, state, round_index)):
            return
