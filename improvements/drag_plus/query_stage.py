"""Query-stage debate: retrieve, let two agents argue, let the Judge decide, refine.

Reproduces the v1.0.1 behaviour including its quirks: the Judge runs every round (even
when its verdict cannot change anything), an unresolved verdict skips to the next round,
Opponent text without an action ends the stage, and optimisation deletes the pool entry
closest to the quoted original by edit distance.

``evidence_store="anchored"`` swaps that deletion for an append-only store in
``evidence/actions.py``; the agent and Judge prompts are unchanged. The prompt builders
render the store's global labels and optional data delimiters.
"""
from __future__ import annotations

from . import prompts
from .adapters import BATCH_ROLE, StageContext
from .evidence import layout
from .evidence.actions import (ACTION_APPLIED, EXIT_DUPLICATE, EXIT_ZERO_NOVELTY,
                               QueryPoolError, maintain_query_pool, prepare_pool,
                               search, sync_pool)
from .judge_decision import JudgeDecision, resolve_judge_decision
from .state import QuestionState

PROPONENT = "Proponent Agent 0"
OPPONENT = "Opponent Agent 0"
AGENTS = (PROPONENT, OPPONENT)


def _budgeted_pool(ctx: StageContext, state: QuestionState, fixed_text: str):
    """Role-first rendering under a budget; the baseline pool without one."""
    return layout.role_pool(ctx, state, fixed_text, "query")


def _agent_prompt(ctx: StageContext, state: QuestionState, agent: str,
                  prefix=None):
    if prefix is not None:
        messages = [prompts.query_agent_system(agent, prefix=prefix),
                    prompts.query_round_user(state.question, state.query_pool,
                                             prefix=prefix)]
    else:
        pool = _budgeted_pool(ctx, state, f"Question: {state.question}\n")
        messages = [prompts.query_agent_system(agent),
                    prompts.query_round_user(state.question, pool)]
    return ctx.backends.render(messages)


def _record_turn(state: QuestionState, round_index: int, agent: str,
                 prompt, output: str) -> None:
    state.emit(f"QueryStage_{agent}_Round{round_index}_InputPrompt", prompt)
    state.emit(f"QueryStage_{agent}_Round{round_index}_Output", output)
    state.query_messages[agent] = [prompt, output]


def _agent_turn(ctx: StageContext, state: QuestionState, round_index: int, agent: str,
                prefix=None) -> None:
    prompt = _agent_prompt(ctx, state, agent, prefix)
    output = ctx.backends.complete(prompt, call=state.call("query_agent", agent))
    _record_turn(state, round_index, agent, prompt, output)


def _agent_round(ctx: StageContext, state: QuestionState, round_index: int,
                 prefix=None) -> None:
    """Both agents of one round in a single 2-prompt call; outputs map by position."""
    prompts = [_agent_prompt(ctx, state, agent, prefix) for agent in AGENTS]
    outputs = ctx.backends.complete_batch(prompts, call=state.call("query_agent", BATCH_ROLE))
    for agent, prompt, output in zip(AGENTS, prompts, outputs):
        _record_turn(state, round_index, agent, prompt, output)


def _run_agents(ctx: StageContext, state: QuestionState, round_index: int,
                prefix=None) -> None:
    if ctx.flags.batch_agents_in_round:
        _agent_round(ctx, state, round_index, prefix)
        return
    for agent in AGENTS:
        _agent_turn(ctx, state, round_index, agent, prefix)


def _judge(ctx: StageContext, state: QuestionState, round_index: int,
           prefix=None) -> JudgeDecision:
    prefix_key = f"QueryStage_Moderator_Round{round_index}"
    outputs = {agent: state.query_messages[agent][1] for agent in state.query_messages}
    pool = state.query_pool
    if prefix is None:
        pool = _budgeted_pool(ctx, state, f"Question: {state.question}\n")
    message = [prompts.query_moderator_system(outputs, state.question, pool,
                                              prefix=prefix)]
    prompt = ctx.backends.render(message)
    output = ctx.backends.complete(prompt, call=state.call("query_judge", "judge"))
    state.emit(f"{prefix_key}_InputPrompt", prompt)
    state.emit(f"{prefix_key}_Output", output)

    def retry() -> str:
        """Re-ask the Judge once with the retry instruction."""
        retry_message = message + [{"role": "assistant", "content": output},
                                   prompts.judge_retry_user()]
        retry_prompt = ctx.backends.render(retry_message)
        retry_output = ctx.backends.complete(
            retry_prompt, call=state.call("query_judge_retry", "judge"))
        state.emit(f"{prefix_key}_RetryInputPrompt", retry_prompt)
        state.emit(f"{prefix_key}_RetryOutput", retry_output)
        return retry_output

    verdict = resolve_judge_decision(output, retry)
    state.emit(f"{prefix_key}_Decision", verdict.decision)
    state.emit(f"{prefix_key}_DecisionStatus", verdict.status)
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


def _round_prefix(ctx: StageContext, state: QuestionState):
    """The evidence-first prefix for this query round, or None on the baseline layout."""
    if ctx.flags.prompt_layout != "evidence_first":
        return None
    prefix, selection = layout.round_prefix(
        state.question, state.query_pool, ctx.flags.context_token_budget, ctx.count_tokens)
    layout.log_selection(state, selection, "query")
    return prefix


def run_query_stage(ctx: StageContext, state: QuestionState) -> None:
    """Fill ``state.query_pool``; the stage ends when the Judge is satisfied."""
    state.stage = "query"
    rounds = ctx.flags.max_query_debate_rounds
    for round_index in range(max(rounds, 1)):
        state.round = round_index
        if round_index == 0:
            prepare_pool(ctx, state)
            question = state.question
            passages = search(ctx, state, question)
            if state.evidence is not None:
                state.evidence.add_query(question.strip(), passages, round_index,
                                         anchored=True)
                sync_pool(ctx, state)
            else:
                state.query_pool[question.strip()] = passages
        if rounds == 0:
            return
        prefix = _round_prefix(ctx, state)
        _run_agents(ctx, state, round_index, prefix)
        if not _continues_after(ctx, state, _judge(ctx, state, round_index, prefix)):
            return
