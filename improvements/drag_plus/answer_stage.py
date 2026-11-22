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
``prompt_layout=evidence_first`` puts one shared header + question + evidence prefix at
the start of every role's system message; the role instructions still follow it.
"""
from __future__ import annotations

from typing import Optional

from . import prompts
from .adapters import BATCH_ROLE, StageContext
from .answers import FINAL_ANSWER_PREFIX
from .evidence import layout
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
    return budget is not None and FINAL_ANSWER_PREFIX.lower() not in output.lower()


def _others(state: QuestionState, agent: str) -> dict:
    return {name: msgs for name, msgs in state.answer_messages.items() if name != agent}


def _debate_text(ctx: StageContext, state: QuestionState, round_index: int,
                 agent: str) -> str:
    return prompts.answer_debate_user(
        _others(state, agent), state.question, round_index, ctx.strategy_qa,
        ctx.flags.concise_agents)["content"]


def _responses(state: QuestionState) -> str:
    return "Agents responses:\n" + "".join(
        f"{name}: {history[-1]['content']}\n"
        for name, history in state.answer_messages.items())


def _fixed_context(ctx: StageContext, state: QuestionState, round_index: int,
                   agent: str) -> str:
    if round_index == 0:
        return f"Question: {state.question}\n"
    return _debate_text(ctx, state, round_index, agent)


def _agent_prompt(ctx: StageContext, state: QuestionState, round_index: int, agent: str,
                  prefix=None):
    question, history = state.question, state.answer_messages
    concise = ctx.flags.concise_agents
    if round_index == 0:
        pool = state.query_pool
        if prefix is None:
            pool = layout.role_pool(ctx, state,
                                    _fixed_context(ctx, state, round_index, agent), "answer")
        history[agent] = [
            prompts.answer_agent_system(agent, pool, ctx.strategy_qa, concise,
                                        prefix=prefix),
            prompts.question_user(question, prefix=prefix)]
    else:
        history[agent].append(
            prompts.answer_debate_user(_others(state, agent), question, round_index,
                                       ctx.strategy_qa, concise, prefix=prefix))
    return ctx.backends.render(history[agent])


def _record_turn(state: QuestionState, round_index: int, agent: str,
                 prompt, output: str, truncated: bool = False) -> None:
    state.emit(f"AnswerStage_{agent}_Round{round_index}_InputPrompt", prompt)
    state.emit(f"AnswerStage_{agent}_Round{round_index}_Output", output)
    state.answer_messages[agent].append({"role": "assistant", "content": output})
    state.truncated = state.truncated or truncated


def _agent_turn(ctx: StageContext, state: QuestionState, round_index: int, agent: str,
                prefix=None) -> None:
    prompt = _agent_prompt(ctx, state, round_index, agent, prefix)
    budget = ctx.flags.agent_max_tokens
    output = ctx.backends.complete(prompt, call=state.call("answer_agent", agent),
                                   **_budget_params(budget))
    _record_turn(state, round_index, agent, prompt, output, _truncated(output, budget))


def _agent_round(ctx: StageContext, state: QuestionState, round_index: int,
                 prefix=None) -> None:
    """Both agents of one round in a single 2-prompt call; outputs map by position."""
    prompts_per_agent = [_agent_prompt(ctx, state, round_index, agent, prefix)
                         for agent in AGENTS]
    budget = ctx.flags.agent_max_tokens
    outputs = ctx.backends.complete_batch(prompts_per_agent,
                                          call=state.call("answer_agent", BATCH_ROLE),
                                          **_budget_params(budget))
    for agent, prompt, output in zip(AGENTS, prompts_per_agent, outputs):
        _record_turn(state, round_index, agent, prompt, output, _truncated(output, budget))


def _run_agents(ctx: StageContext, state: QuestionState, round_index: int,
                prefix=None) -> None:
    if ctx.flags.batch_agents_in_round:
        _agent_round(ctx, state, round_index, prefix)
        return
    for agent in AGENTS:
        _agent_turn(ctx, state, round_index, agent, prefix)


def _moderator_due(ctx: StageContext, round_index: int, rounds: int) -> bool:
    """With moderator_final_only only the last round is moderated (paper Eq. 13)."""
    return not ctx.flags.moderator_final_only or round_index == rounds - 1


def _moderate(ctx: StageContext, state: QuestionState, round_index: int,
              prefix=None) -> None:
    responses = _responses(state)
    pool = state.query_pool
    if prefix is None:
        pool = layout.role_pool(ctx, state,
                                f"Question: {state.question}\n{responses}", "answer")
    message = [prompts.answer_moderator_system(ctx.strategy_qa, prefix=prefix),
               prompts.answer_moderator_user(state.question, pool, responses,
                                             prefix=prefix)]
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


def _answer_only(ctx: StageContext, state: QuestionState, prefix=None) -> None:
    pool = state.query_pool
    if prefix is None:
        pool = layout.role_pool(ctx, state, f"Question: {state.question}\n", "answer")
    message = [prompts.answer_only_system(pool, ctx.strategy_qa, prefix=prefix),
               prompts.question_user(state.question, prefix=prefix)]
    prompt = ctx.backends.render(message)
    budget = ctx.flags.agent_max_tokens
    output = ctx.backends.complete(prompt, call=state.call("answer_agent", "answer_only"),
                                   **_budget_params(budget))
    state.emit("answer_input_prompt", prompt)
    state.truncated = state.truncated or _truncated(output, budget)
    _set_pred(state, output)


def _longest_history(ctx: StageContext, state: QuestionState, round_index: int) -> str:
    """The longest non-prefix text any role will carry this round, for the budget."""
    if round_index == 0:
        return ""
    texts = [_debate_text(ctx, state, round_index, agent) for agent in AGENTS]
    texts.append(_responses(state))
    return max(texts, key=ctx.count_tokens)


def _answer_prefix(ctx: StageContext, state: QuestionState, round_index: int):
    """The evidence-first prefix for this answer round, or None on the baseline layout."""
    if ctx.flags.prompt_layout != "evidence_first":
        return None
    prefix, selection = layout.round_prefix(
        state.question, state.query_pool, ctx.flags.context_token_budget, ctx.count_tokens,
        history_text=_longest_history(ctx, state, round_index))
    layout.log_selection(state, selection, "answer")
    return prefix


def run_answer_stage(ctx: StageContext, state: QuestionState) -> None:
    """Run the answer debate; each round builds the shared prefix when enabled."""
    state.stage = "answer"
    rounds = _effective_rounds(ctx)
    if rounds == 0:
        _answer_only(ctx, state, _answer_prefix(ctx, state, 0))
        return
    for round_index in range(rounds):
        state.round = round_index
        prefix = _answer_prefix(ctx, state, round_index)
        _run_agents(ctx, state, round_index, prefix)
        if _moderator_due(ctx, round_index, rounds):
            _moderate(ctx, state, round_index, prefix)
