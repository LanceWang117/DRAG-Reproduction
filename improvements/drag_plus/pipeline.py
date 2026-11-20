"""Orchestration only: build the per-question state, run the stages, finish ``pred``.

The pipeline holds no per-question data. Each call to ``run_item`` builds its own
``QuestionState``, so questions cannot leak into each other. Evaluation belongs to the
runner (J03); the baseline's closing ``evaluate`` call is deliberately not part of this.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

from .adapters import Backends, StageContext
from .analysis.extract import extract_answer
from .answer_stage import run_answer_stage
from .flags import Flags
from .query_stage import run_query_stage
from .state import QuestionState


class DragPlusPipeline:
    def __init__(self, flags: Flags, generator: Any, retriever: Any, prompt_template: Any,
                 dataset_name: str):
        backends = Backends(generator, retriever, prompt_template)
        self._ctx = StageContext(flags, backends, dataset_name)

    @property
    def flags(self) -> Flags:
        return self._ctx.flags

    def run_item(self, item: Any, key: Optional[Tuple[Any, ...]] = None) -> QuestionState:
        """Answer one dataset item, writing every output to ``item.update_output``."""
        state = QuestionState(item.question, key=key, sink=item.update_output)
        run_query_stage(self._ctx, state)
        state.emit("QueryStage_QueryPool", state.query_pool)
        run_answer_stage(self._ctx, state)
        self._finish(state)
        return state

    def run(self, dataset: Iterable[Any]) -> List[QuestionState]:
        return [self.run_item(item) for item in dataset]

    def _finish(self, state: QuestionState) -> None:
        if self._ctx.flags.answer_extractor == "native":
            return
        state.emit("raw_pred", state.raw_pred)
        state.pred = extract_answer(state.raw_pred, yes_no=self._ctx.strategy_qa)
        state.emit("pred", state.pred)
