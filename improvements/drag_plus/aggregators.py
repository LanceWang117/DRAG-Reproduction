"""Non-debate answer baselines: plurality vote and temperature self-consistency.

Both aggregate answers normalised by E(.) (``answers.unified_pred``), so ``Paris``,
``Paris.`` and ``**Paris**`` are one vote. A tie goes to an answer the evidence text
contains (case-insensitive substring); if that does not decide, the Proponent's answer
(for ``vote``) or the first sample (for ``self_consistency``) wins.

Self-consistency sends ``self_consistency_samples`` copies of one grounded answer prompt
in one ``generate`` call with ``do_sample=True`` and the configured temperature. It is a
separate per-call parameter set: the shared flags are never mutated, and the baseline and
vote calls pass no sampling parameters. Each candidate records its generated-token count,
so a run reports the tokens it spent and two arms can be compared at a matched output
budget.

Also holds ``sequester_verdict``, the relay redaction used by the debate mode: it removes
every ``The answer is: ...`` segment while keeping the reasoning around it.

``aggregator="weighted"`` takes one answer per agent and maximises the fitted linear
score ``w_conf*conf + w_faith*s_faith + w_agr*agr`` (J19). The weights JSON and the
optional MiniCheck scorer are validated in ``grounding.scorer``; the signals of every
distinct answer are emitted for the run record.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import prompts
from .adapters import StageContext
from .answers import unified_pred
from .evidence import layout
from .gates.features import passage_texts
from .grounding.scorer import build_scorer, load_weights, select_weighted
from .query_stage import AGENTS
from .state import QuestionState

SELF_CONSISTENCY_ROLE = "self_consistency"
WITHHELD = "[reasoning withheld]"
_ANSWER_LINE = re.compile(r"the answer is\s*:[^\n]*", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    """One answer with its source and generated-text token count."""
    source: str
    answer: str
    tokens: int


def sequester_verdict(text: str) -> str:
    """Drop each ``The answer is: ...`` segment; keep reasoning, never relay emptiness."""
    kept = "\n".join(_ANSWER_LINE.sub("", line).rstrip() for line in text.splitlines())
    return kept.strip() or WITHHELD


def relayed(ctx: StageContext, state: QuestionState, agent: str, round_index: int) -> dict:
    """Peer histories for one round, with the verdict line withheld when configured."""
    others = {name: history for name, history in state.answer_messages.items()
              if name != agent}
    if not ctx.flags.verdict_sequestration:
        return others
    index = round_index * 2
    hidden = {}
    for name, history in others.items():
        copied = list(history)
        copied[index] = dict(copied[index],
                             content=sequester_verdict(copied[index]["content"]))
        hidden[name] = copied
    return hidden


def _candidate(source: str, raw: str, ctx: StageContext) -> Candidate:
    return Candidate(source, unified_pred(raw, ctx.dataset_name), ctx.count_tokens(raw))


def _budget(ctx: StageContext) -> dict:
    """The shared answer-agent cap, or no parameter at the baseline default."""
    budget = ctx.flags.agent_max_tokens
    return {} if budget is None else {"max_tokens": budget}


def _prefix(ctx: StageContext, state: QuestionState):
    """The evidence-first prefix for round 0, or None on the baseline layout."""
    if ctx.flags.prompt_layout != "evidence_first":
        return None
    prefix, selection = layout.round_prefix(
        state.question, state.query_pool, ctx.flags.context_token_budget,
        ctx.count_tokens)
    layout.log_selection(state, selection, "answer")
    return prefix


def _pool(ctx: StageContext, state: QuestionState, prefix):
    if prefix is not None:
        return state.query_pool
    fixed = f"Question: {state.question}\n"
    return layout.role_pool(ctx, state, fixed, "answer")


def _agent_prompt(ctx: StageContext, state: QuestionState, agent: str, prefix) -> str:
    message = [prompts.answer_agent_system(
                   agent, _pool(ctx, state, prefix), ctx.strategy_qa,
                   ctx.flags.concise_agents, prefix=prefix,
                   grounded=ctx.flags.opponent_mode == "grounded"),
               prompts.question_user(state.question, prefix=prefix)]
    return ctx.backends.render(message)


def _evidence(state: QuestionState) -> str:
    return prompts.format_query_pool(state.query_pool)


def select_answer(candidates: List[Candidate], evidence: str,
                  fallback: Optional[str] = None) -> Tuple[str, str]:
    """Plurality over E(.) answers; ties prefer evidence containment, then ``fallback``."""
    counts = Counter(candidate.answer for candidate in candidates if candidate.answer)
    if not counts:
        return "", "empty"
    best = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == best]
    if len(tied) == 1:
        return tied[0], "plurality"
    supported = [answer for answer in tied if answer.casefold() in evidence.casefold()]
    finalists = supported or tied
    if len(finalists) == 1:
        return finalists[0], "evidence" if supported else "first_seen"
    if fallback is not None and fallback in finalists:
        return fallback, "fallback"
    return finalists[0], "first_seen"


def _record(state: QuestionState, source: str, prompt: str, raw: str) -> None:
    state.emit(f"AnswerStage_{source}_Round0_InputPrompt", prompt)
    state.emit(f"AnswerStage_{source}_Round0_Output", raw)
    state.answer_messages.setdefault(source, []).append(
        {"role": "assistant", "content": raw})


def _finish(state: QuestionState, candidates: List[Candidate], answer: str,
            reason: str, aggregator: str) -> None:
    """Emit the candidates and the aggregate, and set the final answer."""
    total = sum(candidate.tokens for candidate in candidates)
    state.emit("AnswerStage_Aggregator_Candidates",
               [{"source": candidate.source, "answer": candidate.answer,
                 "tokens": candidate.tokens} for candidate in candidates])
    state.emit("AnswerStage_Aggregator_Result",
               {"aggregator": aggregator, "answer": answer, "tie_break": reason,
                "generated_tokens": total})
    state.decisions.append({"stage": "answer", "round": 0, "decision": "aggregator",
                            "aggregator": aggregator, "answer": answer,
                            "tie_break": reason, "generated_tokens": total})
    state.raw_pred = state.pred = answer
    state.emit("pred", answer)


def run_vote(ctx: StageContext, state: QuestionState) -> None:
    """One answer per agent, then a plurality vote with the Proponent as fallback."""
    state.round = 0
    prefix = _prefix(ctx, state)
    candidates = []
    for agent in AGENTS:
        prompt = _agent_prompt(ctx, state, agent, prefix)
        raw = ctx.backends.complete(prompt, call=state.call("answer_agent", agent),
                                    **_budget(ctx))
        _record(state, agent, prompt, raw)
        candidates.append(_candidate(agent, raw, ctx))
    fallback = candidates[0].answer if candidates else None
    answer, reason = select_answer(candidates, _evidence(state), fallback)
    _finish(state, candidates, answer, reason, "vote")


def run_self_consistency(ctx: StageContext, state: QuestionState) -> None:
    """Sample one grounded answer prompt ``m`` times at temperature and vote."""
    state.round = 0
    prompt = _agent_prompt(ctx, state, AGENTS[0], _prefix(ctx, state))
    count = ctx.flags.self_consistency_samples
    params = {"temperature": ctx.flags.self_consistency_temperature, "do_sample": True,
              **_budget(ctx)}
    state.emit("AnswerStage_SelfConsistency_InputPrompt", prompt)
    outputs = ctx.backends.complete_batch(
        [prompt] * count, call=state.call("answer_agent", SELF_CONSISTENCY_ROLE), **params)
    candidates = []
    for index, raw in enumerate(outputs, 1):
        state.emit(f"AnswerStage_SelfConsistency_Sample{index}_Output", raw)
        candidates.append(_candidate(f"sample_{index}", raw, ctx))
    fallback = candidates[0].answer if candidates else None
    answer, reason = select_answer(candidates, _evidence(state), fallback)
    _finish(state, candidates, answer, reason, "self_consistency")


def run_weighted(ctx: StageContext, state: QuestionState) -> None:
    """One answer per agent; maximise the fitted confidence/faithfulness/agreement score."""
    state.round = 0
    weights = load_weights(ctx.flags.aggregator_weights)
    scorer = build_scorer(ctx.flags.faithfulness_scorer, ctx.strategy_qa)
    prefix = _prefix(ctx, state)
    candidates = []
    for agent in AGENTS:
        prompt = _agent_prompt(ctx, state, agent, prefix)
        raw = ctx.backends.complete(prompt, call=state.call("answer_agent", agent),
                                    **_budget(ctx))
        _record(state, agent, prompt, raw)
        candidates.append(_candidate(agent, raw, ctx))
    answer, reason, rows = select_weighted(
        [(candidate.source, candidate.answer) for candidate in candidates],
        passage_texts(state.query_pool), scorer, weights)
    state.emit("AnswerStage_Aggregator_Signals", rows)
    _finish(state, candidates, answer, reason, "weighted")


def run_aggregator(ctx: StageContext, state: QuestionState) -> None:
    """Dispatch a non-moderator aggregator; the moderator path stays in answer_stage."""
    if ctx.flags.aggregator == "vote":
        run_vote(ctx, state)
    elif ctx.flags.aggregator == "self_consistency":
        run_self_consistency(ctx, state)
    elif ctx.flags.aggregator == "weighted":
        run_weighted(ctx, state)
    else:
        raise ValueError(f"not a non-debate aggregator: {ctx.flags.aggregator!r}")
