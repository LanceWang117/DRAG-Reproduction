"""Orchestration only: build per-question state, batch requests, finish ``pred``.

The pipeline holds no per-question data. ``run_batch`` puts one question state in one
worker and lets ``BarrierScheduler`` group its backend requests with the other questions
at the same ``(stage, round, role)``. A group of one delegates to the existing
``Backends`` methods, so ``question_batch_size=1`` keeps the frozen call sequence (T0).
An exception in one worker marks only that state ``failed``; the remaining questions
continue and the runner reports the failed run.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, List, Optional, Tuple

from .adapters import Backends, StageContext
from .answers import unified_pred
from .answer_stage import run_answer_stage
from .flags import Flags
from .gates.escalation import load_model
from .instrument.timing import whitespace_tokens
from .query_stage import run_query_stage
from .scheduling.barrier import (BarrierContractError, BarrierScheduler, GENERATE,
                                 SCORES, SEARCH, Request)
from .state import QuestionState

Entry = Tuple[Any, Optional[Tuple[Any, ...]]]


def _prompt_batch(prompts: Any) -> List[Any]:
    """A single rendered prompt (message list) or an explicit list of prompts."""
    if isinstance(prompts, list) and prompts and not isinstance(prompts[0], dict):
        return list(prompts)
    return [prompts]


class _RequestBackends:
    """Turn each stage backend call into a request and block for its result."""

    def __init__(self, real: Backends, scheduler: BarrierScheduler):
        self._real = real
        self._scheduler = scheduler
        self.generator = real.generator
        self.retriever = real.retriever
        self.prompt_template = real.prompt_template
        self.recorder = real.recorder

    def render(self, messages: list) -> Any:
        return self._real.render(messages)

    def _submit(self, kind: str, call: Any, *, prompts=None, queries=None,
                params=None, labels=None, single=True) -> Any:
        request = Request(kind=kind, stage=getattr(call, "stage", ""),
                          round=getattr(call, "round", 0), role=getattr(call, "role", ""),
                          prompts=list(prompts or []), queries=list(queries or []),
                          params=dict(params or {}), labels=labels, call=call,
                          single=single)
        return self._scheduler.submit(request)

    def complete(self, prompt: Any, *, call: Optional[Any] = None, **params: Any) -> str:
        return self._submit(GENERATE, call, prompts=[prompt], params=params)

    def complete_batch(self, prompts: List[Any], *, call: Optional[Any] = None,
                       **params: Any) -> List[str]:
        return self._submit(GENERATE, call, prompts=prompts, params=params, single=False)

    def complete_with_scores(self, prompts: Any, *, labels, call: Optional[Any] = None,
                             **params: Any):
        """Submit a scored request; the generator must expose ``generate_with_scores``."""
        score = getattr(self._real.generator, "generate_with_scores", None)
        if not callable(score):
            raise TypeError(
                "judge_scoring='logprob' needs a generator with "
                "generate_with_scores(prompts, labels=...)")
        return self._submit(SCORES, call, prompts=_prompt_batch(prompts), params=params,
                            labels=labels, single=False)

    def search(self, query: str, *, call: Optional[Any] = None) -> List[dict]:
        return self._submit(SEARCH, call, queries=[query])

    def batch_search(self, queries: List[str], *, call: Optional[Any] = None):
        return self._submit(SEARCH, call, queries=queries, single=False)

    def execute(self, requests: List[Request]) -> None:
        """Fill one group's results, with one backend call for the whole group."""
        if len(requests) == 1:
            self._execute_one(requests[0])
        elif requests[0].kind == GENERATE:
            self._execute_generate(requests)
        elif requests[0].kind == SCORES:
            self._execute_scores(requests)
        else:
            self._execute_search(requests)

    def _execute_one(self, request: Request) -> None:
        params = dict(request.params)
        if request.kind == GENERATE:
            if request.single:
                request.result = self._real.complete(request.prompts[0],
                                                     call=request.call, **params)
            else:
                request.result = self._real.complete_batch(request.prompts,
                                                          call=request.call, **params)
        elif request.kind == SCORES:
            request.result = self._real.complete_with_scores(
                request.prompts, labels=request.labels, call=request.call, **params)
        elif request.single:
            request.result = self._real.search(request.queries[0], call=request.call)
        else:
            request.result = self._real.batch_search(request.queries, call=request.call)

    def _execute_generate(self, requests: List[Request]) -> None:
        params = requests[0].params
        if any(request.params != params for request in requests):
            raise BarrierContractError("grouped requests disagree on generation params")
        prompts = [prompt for request in requests for prompt in request.prompts]
        outputs = list(self._real.generator.generate(prompts, **params))
        self._split(requests, outputs, self._finish_generate, "prompts")

    def _execute_scores(self, requests: List[Request]) -> None:
        params = requests[0].params
        prompts = [prompt for request in requests for prompt in request.prompts]
        outputs = list(self._real.generator.generate_with_scores(
            prompts, labels=requests[0].labels, **params))
        self._split(requests, outputs, self._finish_scores, "prompts")

    def _execute_search(self, requests: List[Request]) -> None:
        queries = [query for request in requests for query in request.queries]
        results = list(self._real.retriever.batch_search(queries))
        self._split(requests, results, self._finish_search, "queries")

    def _split(self, requests: List[Request], values: List[Any], finish, field_name: str):
        expected = sum(len(getattr(request, field_name)) for request in requests)
        if len(values) != expected:
            raise BarrierContractError(
                f"batch returned {len(values)} results for {expected} {field_name}")
        start = 0
        for request in requests:
            count = len(getattr(request, field_name))
            request.result = finish(request, values[start:start + count])
            start += count

    def _finish_generate(self, request: Request, outputs: List[str]):
        self._record_generation(request, outputs)
        return outputs[0] if request.single else list(outputs)

    def _finish_scores(self, request: Request, outputs: List[Any]):
        recorder = self._real.recorder
        if recorder is not None and request.call is not None:
            recorder.generation_batch(request.call, request.prompts,
                                      lambda: [output.text for output in outputs],
                                      limit=request.params.get("max_tokens"))
        return list(outputs)

    def _finish_search(self, request: Request, results: List[Any]):
        recorder = self._real.recorder
        if recorder is not None and request.call is not None:
            recorder.retrieval(request.call, len(results), lambda: results)
        return results[0] if request.single else list(results)

    def _record_generation(self, request: Request, outputs: List[str]) -> None:
        recorder = self._real.recorder
        if recorder is None or request.call is None:
            return
        limit = request.params.get("max_tokens")
        if request.single and len(request.prompts) == 1:
            recorder.generation(request.call, request.prompts[0], lambda: outputs[0],
                                limit=limit)
        else:
            recorder.generation_batch(request.call, request.prompts, lambda: list(outputs),
                                      limit=limit)


class DragPlusPipeline:
    def __init__(self, flags: Flags, generator: Any, retriever: Any, prompt_template: Any,
                 dataset_name: str, recorder: Any = None,
                 count_tokens: Callable[[str], int] = whitespace_tokens):
        backends = Backends(generator, retriever, prompt_template,
                            recorder if flags.profile_calls else None)
        gate_model = load_model(flags.escalation_model) \
            if flags.escalation_gate == "risk" else None
        self._ctx = StageContext(flags, backends, dataset_name, count_tokens, gate_model)

    @property
    def flags(self) -> Flags:
        return self._ctx.flags

    def run_item(self, item: Any, key: Optional[Tuple[Any, ...]] = None) -> QuestionState:
        """Answer one item through a one-question barrier; re-raise its failure."""
        return self.run_batch([(item, key)], batch_size=1, raise_on_error=True)[0]

    def run(self, dataset: Iterable[Any]) -> List[QuestionState]:
        """Answer the dataset in batches of ``question_batch_size`` (default 1)."""
        return self.run_batch([(item, None) for item in dataset])

    def run_batch(self, entries: Iterable[Entry], batch_size: Optional[int] = None,
                  on_finish: Optional[Callable[[Any, QuestionState], None]] = None,
                  raise_on_error: bool = False) -> List[QuestionState]:
        """Run ``(item, key)`` entries; ``on_finish(item, state)`` runs per finish."""
        entries = list(entries)
        size = self._ctx.flags.question_batch_size if batch_size is None else batch_size
        if size < 1:
            raise ValueError("question_batch_size must be at least 1")
        states: List[QuestionState] = []
        for start in range(0, len(entries), size):
            states.extend(self._run_chunk(entries[start:start + size], on_finish))
        if raise_on_error:
            for state in states:
                if state.status == "failed" and state.exception is not None:
                    raise state.exception
        return states

    def _run_chunk(self, entries: List[Entry],
                   on_finish: Optional[Callable[[Any, QuestionState], None]]
                   ) -> List[QuestionState]:
        real = self._ctx.backends
        scheduler = BarrierScheduler(lambda requests: proxy.execute(requests))
        proxy = _RequestBackends(real, scheduler)
        ctx = replace(self._ctx, backends=proxy)
        states = [QuestionState(item.question, key=key, sink=item.update_output)
                  for item, key in entries]
        by_state = {id(state): item for (item, _), state in zip(entries, states)}
        callback = None if on_finish is None else \
            (lambda state: on_finish(by_state[id(state)], state))
        targets = [(state, lambda state=state: self._run_state(ctx, state))
                   for state in states]
        scheduler.run(targets, callback)
        return states

    def _run_state(self, ctx: StageContext, state: QuestionState) -> None:
        run_query_stage(ctx, state)
        state.emit("QueryStage_QueryPool", state.query_pool)
        run_answer_stage(ctx, state)
        self._finish(state)
        state.status = "done"

    def _finish(self, state: QuestionState) -> None:
        if self._ctx.flags.answer_extractor == "native":
            return
        state.emit("raw_pred", state.raw_pred)
        state.pred = unified_pred(state.raw_pred, self._ctx.dataset_name)
        state.emit("pred", state.pred)
