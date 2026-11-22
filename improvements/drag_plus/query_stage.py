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
from .evidence.store import id_map, latest_refined, register_query
from .judge.action_schema import active_queries
from .judge.actions import apply_opponent_action
from .judge_decision import JudgeDecision, resolve_judge_decision
from .judge.logprob import LABELS, ScoredVerdict, resolve_scored
from .state import QuestionState

PROPONENT = "Proponent Agent 0"
OPPONENT = "Opponent Agent 0"
AGENTS = (PROPONENT, OPPONENT)


def _register_queries(state: QuestionState, source: str) -> None:
    """Give every current plan query a stable ID (J13); existing records keep theirs."""
    for query in active_queries(state):
        register_query(state.queries, query, source, state.round)


def _schema_prompt(ctx: StageContext, state: QuestionState, prefix=None):
    """The schema-mode Opponent prompt: stable-ID map plus baseline evidence pool."""
    pairs = id_map(state.queries, active_queries(state))
    system = prompts.query_opponent_schema_system(pairs, prefix=prefix)
    if prefix is None:
        pool = _budgeted_pool(ctx, state, f"Question: {state.question}\n")
        user = prompts.query_round_user(state.question, pool)
    else:
        user = prompts.query_round_user(state.question, state.query_pool, prefix=prefix)
    return ctx.backends.render([system, user])


def _budgeted_pool(ctx: StageContext, state: QuestionState, fixed_text: str):
    """Role-first rendering under a budget; the baseline pool without one."""
    return layout.role_pool(ctx, state, fixed_text, "query")


def _agent_prompt(ctx: StageContext, state: QuestionState, agent: str,
                  prefix=None):
    if ctx.flags.opponent_action_schema and "Opponent" in agent:
        return _schema_prompt(ctx, state, prefix)
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


def _judge_question(ctx: StageContext, state: QuestionState) -> str:
    """The original question, or the latest active refined query when J13 asks for it."""
    if not ctx.flags.judge_sees_refined_query:
        return state.question
    return latest_refined(state.queries, active_queries(state)) or state.question


def _judge_message(ctx: StageContext, state: QuestionState, prefix, agent_order=None):
    """The Judge's system message for one presentation order (None = baseline)."""
    order = tuple(state.query_messages) if agent_order is None else tuple(agent_order)
    outputs = {agent: state.query_messages[agent][1] for agent in order}
    pool = state.query_pool
    if prefix is None:
        pool = _budgeted_pool(ctx, state, f"Question: {state.question}\n")
    question = _judge_question(ctx, state)
    if prefix is not None and ctx.flags.judge_sees_refined_query:
        prefix = prefix.replace(f"Question: {state.question}",
                                f"Question: {question}", 1)
    return [prompts.query_moderator_system(outputs, question, pool,
                                           prefix=prefix, agent_order=agent_order)]


def _record_verdict(state: QuestionState, round_index: int, verdict) -> None:
    """Emit and append one Judge decision; a log-prob margin rides along as ``score``."""
    prefix_key = f"QueryStage_Moderator_Round{round_index}"
    state.emit(f"{prefix_key}_Decision", verdict.decision)
    state.emit(f"{prefix_key}_DecisionStatus", verdict.status)
    entry = {"stage": "query", "round": round_index, "decision": verdict.decision,
             "status": verdict.status}
    score = getattr(verdict, "score", None)
    if score is not None:
        entry["score"] = score
    state.decisions.append(entry)


def _judge_generative(ctx: StageContext, state: QuestionState, round_index: int,
                      prefix, prefix_key: str) -> JudgeDecision:
    message = _judge_message(ctx, state, prefix)
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
    _record_verdict(state, round_index, verdict)
    return verdict


def _judge_logprob(ctx: StageContext, state: QuestionState, round_index: int,
                   prefix, prefix_key: str) -> ScoredVerdict:
    """One scored call; order balancing sends both presentations in the same batch."""
    order = tuple(AGENTS)
    orders = [order] + ([order[::-1]] if ctx.flags.judge_order_balance else [])
    rendered = [ctx.backends.render(_judge_message(ctx, state, prefix, agent_order))
                for agent_order in orders]
    scored = ctx.backends.complete_with_scores(
        rendered, labels=LABELS, call=state.call("query_judge", "judge"))
    for index, (prompt, output) in enumerate(zip(rendered, scored)):
        suffix = "" if index == 0 else f"_Order{index + 1}"
        state.emit(f"{prefix_key}_InputPrompt{suffix}", prompt)
        state.emit(f"{prefix_key}_Output{suffix}", output.text)
    verdict = resolve_scored(scored, ctx.flags.judge_tau, ctx.flags.judge_order_balance)
    state.judge_score = verdict.score
    if verdict.score is not None:
        state.emit(f"{prefix_key}_Score", verdict.score)
    if verdict.status != "valid_logprob":
        state.emit(f"{prefix_key}_Fallback", verdict.status)
        state.decisions.append({"stage": "query", "round": round_index,
                                "decision": "judge_fallback",
                                "status": verdict.status})
    _record_verdict(state, round_index, verdict)
    return verdict


def _judge(ctx: StageContext, state: QuestionState, round_index: int,
           prefix=None) -> JudgeDecision | ScoredVerdict:
    prefix_key = f"QueryStage_Moderator_Round{round_index}"
    if ctx.flags.judge_scoring == "logprob":
        return _judge_logprob(ctx, state, round_index, prefix, prefix_key)
    return _judge_generative(ctx, state, round_index, prefix, prefix_key)


def _continues_after(ctx: StageContext, state: QuestionState, verdict: JudgeDecision) -> bool:
    """True when the debate goes on to another round after this verdict."""
    if verdict.decision is None:
        return True
    if verdict.decision == "Proponent":
        return False
    output = state.query_messages[OPPONENT][1]
    if ctx.flags.opponent_action_schema:
        outcome = apply_opponent_action(ctx, state, output)
    else:
        outcome = maintain_query_pool(ctx, state, output)
        source = ("optimize" if "Query Optimization:" in output else
                  "expand" if "Query Expansion:" in output else "refined")
        _register_queries(state, source)
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
            _register_queries(state, "original")
        if rounds == 0:
            return
        prefix = _round_prefix(ctx, state)
        _run_agents(ctx, state, round_index, prefix)
        if not _continues_after(ctx, state, _judge(ctx, state, round_index, prefix)):
            return
