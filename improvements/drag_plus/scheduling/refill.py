"""Refill-window scheduling over the cross-question barrier (J16).

One window slot is one active question. A *step* is one barrier: every live question
blocks on a backend request, the scheduler dispatches the step, and each finished
question frees its slot for the next pending question in manifest order. Stage code is
unchanged, so a question's prompts and outputs do not depend on its neighbours; only
the composition of each batch changes. Orca and vLLM apply the same idea one level
down, scheduling model iterations instead of whole requests (J16 research note).

``Request`` and ``SchedulerStats`` live here with the window because the barrier and
the pipeline both use them; ``scheduling.barrier`` re-exports them for existing callers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

GENERATE = "generate"
SCORES = "scores"
SEARCH = "search"

Target = Tuple[Any, Any]


def _freeze(value: Any) -> Any:
    """Make a params/labels value hashable for the group key."""
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass
class Request:
    """One backend call owned by one question; the scheduler fills exactly one result."""

    kind: str
    stage: str
    round: int
    role: str
    prompts: List[Any] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    labels: Any = None
    call: Any = None
    single: bool = True
    result: Any = None
    error: Optional[BaseException] = None
    done: bool = False

    def key(self) -> tuple:
        return (self.kind, self.stage, self.round, self.role,
                _freeze(self.params), _freeze(self.labels))


@dataclass
class SchedulerStats:
    """Mechanical evidence: one barrier and one call per dispatched group."""

    barriers: int = 0
    generate_calls: int = 0
    score_calls: int = 0
    search_calls: int = 0


class BarrierError(RuntimeError):
    """A request was submitted outside a worker or a group result was malformed."""


class BarrierContractError(BarrierError):
    """A backend returned a different number of results than prompts or queries."""


class RefillWindow:
    """Admission policy: keep ``size`` questions live, admitting in manifest order.

    ``fill(live)`` returns the next ``size - live`` ``(state, target)`` pairs and marks
    ``exhausted`` once the pending queue is empty. ``record(active)`` is called by the
    scheduler once per dispatched step so ``mean_occupancy`` can be reported as a
    fraction of the window. A size below 1 is a programming error, not a flag error:
    the flag validator already enforces ``question_batch_size >= 1``.
    """

    def __init__(self, targets: Iterable[Target], size: int):
        if size < 1:
            raise ValueError("refill window must be at least 1")
        self.size = size
        self._targets: Deque[Target] = deque(targets)
        self.exhausted = False
        self.admissions = 0
        self.active_steps: List[int] = []

    def fill(self, live: int) -> List[Target]:
        """Return up to ``size - live`` pending targets, in admission order."""
        room = max(0, self.size - live)
        targets: List[Target] = []
        while self._targets and len(targets) < room:
            targets.append(self._targets.popleft())
        self.exhausted = not self._targets
        self.admissions += len(targets)
        return targets

    def record(self, active: int) -> None:
        """Record the number of questions in one dispatched step."""
        self.active_steps.append(active)

    @property
    def mean_occupancy(self) -> float:
        """Mean active questions per step, as a fraction of the window (0 when idle)."""
        if not self.active_steps:
            return 0.0
        return sum(self.active_steps) / (len(self.active_steps) * self.size)


def targets_for(states: Sequence[Any], run: Any) -> List[Target]:
    """Pair each state with a zero-argument worker target that runs it."""
    return [(state, lambda state=state: run(state)) for state in states]


def gather_search(requests: List[Request], retriever: Any, cap: int) -> int:
    """Fill every ready request from one gathered, cap-sized ``batch_search`` sequence.

    All queries of the step are flattened in request order, so results stitch back to
    the owning requests without per-request calls. Returns the number of backend calls
    (more than one only when the gathered queries exceed ``cap``) so the scheduler's
    stats stay truthful. A row-count mismatch is a contract error, never a guess.
    """
    collected: Dict[int, List[Any]] = {id(request): [] for request in requests}
    pairs = [(request, query) for request in requests for query in request.queries]
    calls = 0
    for start in range(0, len(pairs), cap):
        chunk = pairs[start:start + cap]
        values = list(retriever.batch_search([query for _, query in chunk]))
        if len(values) != len(chunk):
            raise BarrierContractError(
                f"batch_search returned {len(values)} results for {len(chunk)} queries")
        for (request, _), value in zip(chunk, values):
            collected[id(request)].append(value)
        calls += 1
    for request in requests:
        request.result = collected[id(request)]
    return calls
