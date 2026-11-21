"""Answer stage: a closed-book and a grounded agent debate, a moderator picks ``pred``.

Reproduces the v1.0.1 behaviour: the moderator runs after every round and each round
overwrites ``pred``, so the last round wins; round ``r`` shows each agent the other's
round ``r-1`` answer (history index ``2r``).

``moderator_final_only`` runs the moderator once, after the last round, with the same
last-round prompt; the rounds 0..R-2 ``AnswerStage_Moderator_Round*`` keys are absent.
"""
from __future__ import annotations

from . import prompts
from .adapters import StageContext
from .query_stage import AGENTS
from .state import QuestionState


def _agent_prompt(ctx: StageContext, state: QuestionState, round_index: int, agent: str):
    question, history = state.question, state.answer_messages
    if round_index == 0:
        history[agent] = [
            prompts.answer_agent_system(agent, state.query_pool, ctx.strategy_qa),
            prompts.question_user(question)]
    else:
        others = {name: msgs for name, msgs in history.items() if name != agent}
        history[agent].append(
            prompts.answer_debate_user(others, question, round_index, ctx.strategy_qa))
    return ctx.backends.render(history[agent])


def _agent_turn(ctx: StageContext, state: QuestionState, round_index: int, agent: str) -> None:
    prompt = _agent_prompt(ctx, state, round_index, agent)
    output = ctx.backends.complete(prompt, call=state.call("answer_agent", agent))
    state.emit(f"AnswerStage_{agent}_Round{round_index}_InputPrompt", prompt)
    state.emit(f"AnswerStage_{agent}_Round{round_index}_Output", output)
    state.answer_messages[agent].append({"role": "assistant", "content": output})


def _moderator_due(ctx: StageContext, round_index: int, rounds: int) -> bool:
    """With moderator_final_only only the last round is moderated (paper Eq. 13)."""
    return not ctx.flags.moderator_final_only or round_index == rounds - 1


def _moderate(ctx: StageContext, state: QuestionState, round_index: int) -> None:
    responses = "Agents responses:\n" + "".join(
        f"{name}: {history[-1]['content']}\n" for name, history in state.answer_messages.items())
    message = [prompts.answer_moderator_system(ctx.strategy_qa),
               prompts.answer_moderator_user(state.question, state.query_pool, responses)]
    prompt = ctx.backends.render(message)
    output = ctx.backends.complete(prompt, call=state.call("answer_moderator", "moderator"))
    state.emit(f"AnswerStage_Moderator_Round{round_index}_InputPrompt", prompt)
    state.emit(f"AnswerStage_Moderator_Round{round_index}_Output", output)
    _set_pred(state, output)


def _set_pred(state: QuestionState, output: str) -> None:
    state.emit("pred", output)
    state.pred = state.raw_pred = output


def _answer_only(ctx: StageContext, state: QuestionState) -> None:
    message = [prompts.answer_only_system(state.query_pool, ctx.strategy_qa),
               prompts.question_user(state.question)]
    prompt = ctx.backends.render(message)
    output = ctx.backends.complete(prompt, call=state.call("answer_agent", "answer_only"))
    state.emit("answer_input_prompt", prompt)
    _set_pred(state, output)


def run_answer_stage(ctx: StageContext, state: QuestionState) -> None:
    state.stage = "answer"
    rounds = ctx.flags.max_answer_debate_rounds
    if rounds == 0:
        _answer_only(ctx, state)
        return
    for round_index in range(rounds):
        state.round = round_index
        for agent in AGENTS:
            _agent_turn(ctx, state, round_index, agent)
        if _moderator_due(ctx, round_index, rounds):
            _moderate(ctx, state, round_index)
