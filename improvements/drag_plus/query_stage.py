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
from .adapters import StageContext
from .judge_decision import JudgeDecision, resolve_judge_decision
from .state import QuestionState

PROPONENT = "Proponent Agent 0"
OPPONENT = "Opponent Agent 0"
AGENTS = (PROPONENT, OPPONENT)


class QueryPoolError(ValueError):
    """The Opponent's proposal could not be applied to the query pool."""


def find_most_similar_key(pool: dict, target: str) -> Optional[str]:
    best_key, best = None, float("inf")
    for key in pool:
        distance = Levenshtein.distance(key, target)
        if distance < best:
            best, best_key = distance, key
    return best_key


def _apply_action(ctx: StageContext, pool: dict, text: str) -> Optional[dict]:
    if "Query Optimization:" in text:
        instruction = text.split("Query Optimization:")[1].strip()
        if "->" in instruction:
            parts = instruction.split("->")
            original, new_query = parts[0].strip(), parts[1].strip()
        else:
            original = new_query = instruction
        pool.pop(find_most_similar_key(pool, original))
        pool[new_query] = ctx.backends.search(new_query)
    elif "Query Expansion:" in text:
        new_query = text.split("Query Expansion:")[1].strip()
        pool[new_query] = ctx.backends.search(new_query)
    else:
        return None
    return pool


def maintain_query_pool(ctx: StageContext, pool: dict, opponent_output: str) -> Optional[dict]:
    """Apply the Opponent's action in place; None means it proposed no action."""
    try:
        return _apply_action(ctx, pool, opponent_output)
    except Exception as error:
        raise QueryPoolError(
            f"Error in maintaining the query pool for Opponent output {opponent_output!r}"
        ) from error


def _agent_turn(ctx: StageContext, state: QuestionState, round_index: int, agent: str) -> None:
    messages = [prompts.query_agent_system(agent),
                prompts.query_round_user(state.question, state.query_pool)]
    prompt = ctx.backends.render(messages)
    output = ctx.backends.complete(prompt)
    state.emit(f"QueryStage_{agent}_Round{round_index}_InputPrompt", prompt)
    state.emit(f"QueryStage_{agent}_Round{round_index}_Output", output)
    state.query_messages[agent] = [prompt, output]


def _judge(ctx: StageContext, state: QuestionState, round_index: int) -> JudgeDecision:
    prefix = f"QueryStage_Moderator_Round{round_index}"
    outputs = {agent: state.query_messages[agent][1] for agent in state.query_messages}
    message = [prompts.query_moderator_system(outputs, state.question, state.query_pool)]
    prompt = ctx.backends.render(message)
    output = ctx.backends.complete(prompt)
    state.emit(f"{prefix}_InputPrompt", prompt)
    state.emit(f"{prefix}_Output", output)

    def retry() -> str:
        retry_message = message + [{"role": "assistant", "content": output},
                                   prompts.judge_retry_user()]
        retry_prompt = ctx.backends.render(retry_message)
        retry_output = ctx.backends.complete(retry_prompt)
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
    updated = maintain_query_pool(ctx, state.query_pool, state.query_messages[OPPONENT][1])
    return updated is not None


def run_query_stage(ctx: StageContext, state: QuestionState) -> None:
    """Fill ``state.query_pool``; the stage ends when the Judge is satisfied."""
    state.stage = "query"
    rounds = ctx.flags.max_query_debate_rounds
    for round_index in range(max(rounds, 1)):
        state.round = round_index
        if round_index == 0:
            question = state.question
            state.query_pool[question.strip()] = ctx.backends.search(question)
        if rounds == 0:
            return
        for agent in AGENTS:
            _agent_turn(ctx, state, round_index, agent)
        if not _continues_after(ctx, state, _judge(ctx, state, round_index)):
            return
