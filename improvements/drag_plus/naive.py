"""The Naive RAG arm: one retrieval, one generation, as FlashRAG's SequentialPipeline.

For a single question that is ``batch_search([q])``, ``get_string(question=,
retrieval_result=)`` and one ``generate`` call, followed by the extractor. ``raw_pred`` is
always written, as ``single_agent_pred_parse`` does.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from .adapters import Backends, StageContext
from .answers import native_pred, unified_pred
from .flags import Flags
from .state import QuestionState


class NaivePipeline:
    def __init__(self, flags: Flags, generator: Any, retriever: Any, prompt_template: Any,
                 dataset_name: str, recorder: Any = None):
        backends = Backends(generator, retriever, prompt_template,
                            recorder if flags.profile_calls else None)
        if not callable(getattr(retriever, "batch_search", None)):
            raise TypeError("retriever does not implement batch_search")
        self._ctx = StageContext(flags, backends, dataset_name)

    def run_item(self, item: Any, key: Optional[Tuple[Any, ...]] = None) -> QuestionState:
        ctx, state = self._ctx, QuestionState(item.question, key=key, sink=item.update_output)
        docs = ctx.backends.batch_search([item.question],
                                         call=state.call("retrieval", "retriever"))[0]
        state.emit("retrieval_result", docs)
        prompt = ctx.backends.prompt_template.get_string(
            question=item.question, retrieval_result=docs)
        state.emit("prompt", prompt)
        raw = ctx.backends.complete(prompt, call=state.call("answer_agent", "naive"))
        state.emit("pred", raw)
        self._finish(state, raw)
        return state

    def _finish(self, state: QuestionState, raw: str) -> None:
        extractor = self._ctx.flags.answer_extractor
        state.raw_pred = raw
        state.emit("raw_pred", raw)
        if extractor == "native":
            state.pred = native_pred("naive", raw)
        else:
            state.pred = unified_pred(raw, self._ctx.dataset_name)
        state.emit("pred", state.pred)
        state.stage, state.status = "answer", "done"
