"""Per-question state record: the only mutable data a pipeline run owns.

Nothing about a question survives on the pipeline object. Stages read and write this
record, and every ``update_output`` write goes through ``emit`` so it is both forwarded
to the dataset item and kept in order for transcripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .instrument.timing import CallInfo

Sink = Callable[[str, Any], None]


@dataclass
class QuestionState:
    question: str
    key: Optional[Tuple[Any, ...]] = None
    query_pool: Dict[str, list] = field(default_factory=dict)
    query_messages: Dict[str, list] = field(default_factory=dict)
    answer_messages: Dict[str, list] = field(default_factory=dict)
    decisions: List[dict] = field(default_factory=list)
    outputs: List[Tuple[str, Any]] = field(default_factory=list)
    stage: str = "query"
    round: int = 0
    status: str = "active"
    pred: Optional[str] = None
    raw_pred: Optional[str] = None
    truncated: bool = False
    calls: List[dict] = field(default_factory=list)
    sink: Optional[Sink] = None

    def call(self, stage: str, role: str) -> CallInfo:
        """Describe the next backend call; timing rows land in ``self.calls``."""
        return CallInfo(stage, self.round, role, self.calls)

    def emit(self, key: str, value: Any) -> None:
        """Record one output and forward it to the sink (the dataset item)."""
        self.outputs.append((key, value))
        if self.sink is not None:
            self.sink(key, value)
