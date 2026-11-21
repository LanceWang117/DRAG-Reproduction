"""Answer stage: a closed-book and a grounded agent debate, a moderator picks ``pred``.

Reproduces the v1.0.1 behaviour: the moderator runs after every round and each round
overwrites ``pred``, so the last round wins; round ``r`` shows each agent the other's
round ``r-1`` answer (history index ``2r``).

``moderator_final_only`` runs the moderator once, after the last round, with the same
last-round prompt; the rounds 0..R-2 ``AnswerStage_Moderator_Round*`` keys are absent.
``batch_agents_in_round`` sends both agents' prompts of a round in one 2-prompt call.
``response_rounds`` overrides the answer-stage count; ``agent_max_tokens`` and
``moderator_max_tokens`` are passed per call, and ``concise_agents`` shortens only the
agent instruction. A budgeted output without the answer marker sets ``state.truncated``.
"""
from __future__ import annotations

from typing import Optional

from . import prompts
from .adapters import BATCH_ROLE, StageContext
from .answers import FINAL_ANSWER_PREFIX
from .query_stage import AGENTS
from .state import QuestionState


def _effective_rounds(ctx: StageContext) -> int:
    """The explicit response budget, or the baseline ``max_answer_debate_rounds``."""
    override = ctx.flags.response_rounds
    return ctx.flags.max_answer_debate_rounds if override is None else override


def _budget_params(budget: Optional[int]) -> dict:
    """Per-call params; a null budget keeps the baseline call parameter-free."""
    return {} if budget is None else {"max_tokens": budget}


def _truncated(output: str, budget: Optional[int]) -> bool:
    """A budgeted answer without its marker is presumed cut off before the answer."""
    return budget is not None and FINAL_ANSWER_PREFIX not in output


def _agent_prompt(ctx: StageContext, state: QuestionState, round_index: int, agent: str):
    question, history = state.question, state.answer_messages
    concise = ctx.flags.concise_agents
    if round_index == 0:
        history[agent] = [
            prompts.answer_agent_system(agent, state.query_pool, ctx.strategy_qa, concise),
            prompts.question_user(question)]
    else:
        others = {name: msgs for name, msgs in history.items() if name != agent}
        history[agent].append(
            prompts.answer_debate_user(others, question, round_index, ctx.strategy_qa,
                                       concise))
    return ctx.backends.render(history[agent])


def _record_turn(state: QuestionState, round_index: int, agent: str,
                 prompt, output: str, truncated: bool = False) -> None:
    state.emit(f"AnswerStage_{agent}_Round{round_index}_InputPrompt", prompt)
    state.emit(f"AnswerStage_{agent}_Round{round_index}_Output", output)
    state.answer_messages[agent].append({"role": "assistant", "content": output})
    state.truncated = state.truncated or truncated


def _agent_turn(ctx: StageContext, state: QuestionState, round_index: int, agent: str) -> None:
    prompt = _agent_prompt(ctx, state, round_index, agent)
    budget = ctx.flags.agent_max_tokens
    output = ctx.backends.complete(prompt, call=state.call("answer_agent", agent),
                                   **_budget_params(budget))
    _record_turn(state, round_index, agent, prompt, output, _truncated(output, budget))


def _agent_round(ctx: StageContext, state: QuestionState, round_index: int) -> None:
    """Both agents of one round in a single 2-prompt call; outputs map by position."""
    prompts_per_agent = [_agent_prompt(ctx, state, round_index, agent) for agent in AGENTS]
    budget = ctx.flags.agent_max_tokens
    outputs = ctx.backends.complete_batch(prompts_per_agent,
                                          call=state.call("answer_agent", BATCH_ROLE),
                                          **_budget_params(budget))
    for agent, prompt, output in zip(AGENTS, prompts_per_agent, outputs):
        _record_turn(state, round_index, agent, prompt, output, _truncated(output, budget))


def _run_agents(ctx: StageContext, state: QuestionState, round_index: int) -> None:
    if ctx.flags.batch_agents_in_round:
        _agent_round(ctx, state, round_index)
        return
    for agent in AGENTS:
        _agent_turn(ctx, state, round_index, agent)


def _moderator_due(ctx: StageContext, round_index: int, rounds: int) -> bool:
    """With moderator_final_only only the last round is moderated (paper Eq. 13)."""
    return not ctx.flags.moderator_final_only or round_index == rounds - 1


def _moderate(ctx: StageContext, state: QuestionState, round_index: int) -> None:
    responses = "Agents responses:\n" + "".join(
        f"{name}: {history[-1]['content']}\n" for name, history in state.answer_messages.items())
    message = [prompts.answer_moderator_system(ctx.strategy_qa),
               prompts.answer_moderator_user(state.question, state.query_pool, responses)]
    prompt = ctx.backends.render(message)
    budget = ctx.flags.moderator_max_tokens
    output = ctx.backends.complete(prompt, call=state.call("answer_moderator", "moderator"),
                                   **_budget_params(budget))
    state.emit(f"AnswerStage_Moderator_Round{round_index}_InputPrompt", prompt)
    state.emit(f"AnswerStage_Moderator_Round{round_index}_Output", output)
    state.truncated = state.truncated or _truncated(output, budget)
    _set_pred(state, output)


def _set_pred(state: QuestionState, output: str) -> None:
    state.emit("pred", output)
    state.pred = state.raw_pred = output


def _answer_only(ctx: StageContext, state: QuestionState) -> None:
    message = [prompts.answer_only_system(state.query_pool, ctx.strategy_qa),
               prompts.question_user(state.question)]
    prompt = ctx.backends.render(message)
    budget = ctx.flags.agent_max_tokens
    output = ctx.backends.complete(prompt, call=state.call("answer_agent", "answer_only"),
                                   **_budget_params(budget))
    state.emit("answer_input_prompt", prompt)
    state.truncated = state.truncated or _truncated(output, budget)
    _set_pred(state, output)


def run_answer_stage(ctx: StageContext, state: QuestionState) -> None:
    state.stage = "answer"
    rounds = _effective_rounds(ctx)
    if rounds == 0:
        _answer_only(ctx, state)
        return
    for round_index in range(rounds):
        state.round = round_index
        _run_agents(ctx, state, round_index)
        if _moderator_due(ctx, round_index, rounds):
            _moderate(ctx, state, round_index)
