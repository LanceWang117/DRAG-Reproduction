"""IRCoT and Self-Ask comparison arms (baseline comparison only).

Both use the same injected backends and question state as the team pipelines, so the
offline fake backend can construct and run them in tests. Their PC outputs are structural
evidence, never results: the real comparison belongs on the A100 (E-jobs).
"""
from __future__ import annotations

from ..adapters import Backends, StageContext
from ..analysis.extract import extract_answer
from ..answers import unified_pred
from ..state import QuestionState
from .execute import _resolved
from .plan import Plan, PlanError, parse_plan, planner_messages, single_hop_plan


class _ComparisonPipeline:
    """Shared glue: build the context, render a Naive prompt and finish a state."""

    def __init__(self, flags, generator, retriever, prompt_template, dataset_name,
                 recorder=None):
        backends = Backends(generator, retriever, prompt_template,
                            recorder if flags.profile_calls else None)
        self._ctx = StageContext(flags, backends, dataset_name)

    def _start(self, item, key):
        return QuestionState(item.question, key=key, sink=item.update_output)

    def _prompt(self, question, docs):
        return self._ctx.backends.prompt_template.get_string(
            question=question, retrieval_result=list(docs))

    def _finish(self, state, raw):
        state.raw_pred = raw
        state.pred = unified_pred(raw, self._ctx.dataset_name)
        state.stage, state.status = "answer", "done"

    def run_item(self, item, key=None):
        raise NotImplementedError


class IRCoTPipeline(_ComparisonPipeline):
    """Interleave one reasoning generation with one retrieval, bounded at two rounds."""

    def run_item(self, item, key=None):
        """Answer one question by alternating reasoning and retrieval."""
        state, ctx = self._start(item, key), self._ctx
        docs = ctx.backends.search(item.question, call=state.call("retrieval", "retriever"))
        raw, rounds = "", max(1, min(2, ctx.flags.max_query_debate_rounds))
        for index in range(rounds):
            state.round = index
            raw = ctx.backends.complete(self._prompt(item.question, docs),
                                        call=state.call("answer_agent", "ircot"))
            state.emit(f"IRCoT_Round{index}", raw)
            if index + 1 < rounds:
                query = extract_answer(raw, False).strip() or item.question
                docs = docs + ctx.backends.search(
                    query, call=state.call("retrieval", "retriever"))
        self._finish(state, raw)
        return state


class SelfAskPipeline(_ComparisonPipeline):
    """Plan, answer each hop with retrieval, then use the last hop as the answer."""

    def run_item(self, item, key=None):
        """Plan, answer each hop with retrieval, and return the last hop's answer."""
        state, ctx = self._start(item, key), self._ctx
        plan = self._plan(state)
        state.emit("SelfAsk_Plan", plan.as_dict())
        bindings, raw = {}, ""
        for step in plan.steps:
            query = _resolved(step.text, bindings) or state.question
            docs = ctx.backends.search(query, call=state.call("retrieval", "retriever"))
            raw = ctx.backends.complete(self._prompt(query, docs),
                                        call=state.call("answer_agent", "selfask"))
            bindings[step.qid] = extract_answer(raw, False).strip() or raw
        self._finish(state, raw)
        return state

    def _plan(self, state) -> Plan:
        try:
            raw = self._ctx.backends.complete(
                self._ctx.backends.render(planner_messages(state.question)),
                call=state.call("query_agent", "planner"))
            return parse_plan(raw)
        except PlanError:
            return single_hop_plan(state.question)
