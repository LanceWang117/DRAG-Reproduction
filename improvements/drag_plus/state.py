"""Per-question state record: the only mutable data a pipeline run owns.

Nothing about a question survives on the pipeline object. Stages read and write this
record, and every ``update_output`` write goes through ``emit`` so it is both forwarded
to the dataset item and kept in order for transcripts. A barrier worker owns one state
at a time; ``status`` becomes ``failed`` with a traceback summary in ``error`` when that
question raises, without touching the others. ``queries`` is the stable query
trail (J13): ``{qid, text, source, round}`` in first-seen order, where a numeric ID is
never reused even after its query leaves the active pool.
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
    evidence: Optional[Any] = None
    queries: List[dict] = field(default_factory=list)
    query_messages: Dict[str, list] = field(default_factory=dict)
    answer_messages: Dict[str, list] = field(default_factory=dict)
    decisions: List[dict] = field(default_factory=list)
    judge_score: Optional[float] = None
    gate_features: Optional[dict] = None
    gate_risk: Optional[float] = None
    sufficiency_score: Optional[float] = None
    router_route: Optional[str] = None
    router_features: Optional[dict] = None
    router_cost: Optional[float] = None
    conformal_qhat: Optional[float] = None
    conformal_set: Optional[List[str]] = None
    conformal_abstained: bool = False
    conformal_escalated: bool = False
    outputs: List[Tuple[str, Any]] = field(default_factory=list)
    stage: str = "query"
    round: int = 0
    status: str = "active"
    error: Optional[str] = None
    exception: Optional[BaseException] = None
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
