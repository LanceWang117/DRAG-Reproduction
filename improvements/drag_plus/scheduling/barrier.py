"""Cross-question barrier scheduler.

A stage call that needs the generator or retriever submits a ``Request`` and blocks.
One worker thread runs each question, so a step is a pure function of that question's
state; the scheduler itself holds no question data. At every barrier it waits until all
live workers are blocked, groups their requests by ``(kind, stage, round, role, params,
labels)``, executes exactly one backend call per group, and hands each request its own
slice of the result. Finished questions leave the barrier; the rest continue.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GENERATE = "generate"
SCORES = "scores"
SEARCH = "search"


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


class _Aborted(BaseException):
    """The scheduler is shutting down; workers must not consume a partial result."""


@dataclass
class _Worker:
    state: Any
    target: Callable[[], None]
    index: int
    request: Optional[Request] = None
    done: bool = False
    notified: bool = False


class BarrierScheduler:
    """Run workers in lockstep and batch their backend requests at each barrier."""

    def __init__(self, execute: Callable[[List[Request]], None]):
        self._execute = execute
        self._lock = threading.Condition()
        self._local = threading.local()
        self._workers: List[_Worker] = []
        self._shutdown: Optional[BaseException] = None
        self.stats = SchedulerStats()

    def run(self, workers: Sequence[Tuple[Any, Callable[[], None]]],
            on_done: Optional[Callable[[Any], None]] = None) -> None:
        """Run ``(state, target)`` pairs, calling ``on_done(state)`` when each finishes."""
        self._workers = [_Worker(state, target, index)
                         for index, (state, target) in enumerate(workers)]
        threads = [threading.Thread(target=self._worker_main, args=(worker,), daemon=True)
                   for worker in self._workers]
        for thread in threads:
            thread.start()
        try:
            self._loop(on_done)
        except _Aborted:
            pass
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            for thread in threads:
                thread.join(timeout=5)
        if self._shutdown is not None:
            raise self._shutdown

    def submit(self, request: Request) -> Any:
        """Block the calling worker until the scheduler has filled ``request``."""
        worker = getattr(self._local, "worker", None)
        if worker is None:
            raise BarrierError("backend request outside a scheduled question")
        with self._lock:
            if self._shutdown is not None:
                raise _Aborted
            worker.request = request
            self._lock.notify_all()
            while not request.done:
                if self._shutdown is not None:
                    raise _Aborted
                self._lock.wait()
            worker.request = None
        if request.error is not None:
            raise request.error
        return request.result

    def _loop(self, on_done: Optional[Callable[[Any], None]]) -> None:
        while True:
            batch, finished = self._wait_batch()
            for state in finished:
                if on_done is not None:
                    on_done(state)
            if batch is not None:
                self._dispatch(batch)
            elif self._shutdown is not None:
                raise _Aborted
            elif all(worker.done for worker in self._workers):
                return

    def _wait_batch(self):
        """Wait for a full barrier; return pending requests and newly finished states."""
        with self._lock:
            while True:
                finished = self._finished()
                if self._shutdown is not None:
                    return None, finished
                if finished:
                    return None, finished
                live = [worker for worker in self._workers if not worker.done]
                if not live:
                    return None, finished
                if self._all_blocked(live):
                    return [worker.request for worker in live], finished
                self._lock.wait()

    def _finished(self) -> List[Any]:
        """Drain the states whose workers finished since the last barrier."""
        states = []
        for worker in self._workers:
            if worker.done and not worker.notified:
                worker.notified = True
                states.append(worker.state)
        return states

    @staticmethod
    def _all_blocked(live: List[_Worker]) -> bool:
        """True when every live worker has submitted an unfinished request."""
        for worker in live:
            if worker.request is None or worker.request.done:
                return False
        return True

    def _dispatch(self, requests: List[Request]) -> None:
        groups: Dict[tuple, List[Request]] = {}
        for request in requests:
            groups.setdefault(request.key(), []).append(request)
        self.stats.barriers += 1
        for group in groups.values():
            self._run_group(group)
        with self._lock:
            for request in requests:
                request.done = True
            self._lock.notify_all()

    def _run_group(self, group: List[Request]) -> None:
        try:
            self._execute(group)
        except BarrierContractError:
            raise
        except Exception as error:
            if len(group) == 1:
                group[0].error = error
                return
            for request in group:
                self._run_group([request])
            return
        if group[0].kind == GENERATE:
            self.stats.generate_calls += 1
        elif group[0].kind == SEARCH:
            self.stats.search_calls += 1
        else:
            self.stats.score_calls += 1

    def _worker_main(self, worker: _Worker) -> None:
        self._local.worker = worker
        try:
            worker.target()
        except _Aborted:
            return
        except BaseException as error:
            if isinstance(error, Exception):
                worker.state.status = "failed"
                worker.state.error = traceback.format_exc(limit=3)
                worker.state.exception = error
                worker.state.pred = worker.state.pred or ""
                worker.state.emit("error", worker.state.error)
            else:
                self._abort(error)
        finally:
            with self._lock:
                worker.done = True
                worker.request = None
                self._lock.notify_all()

    def _abort(self, error: BaseException) -> None:
        with self._lock:
            if self._shutdown is None:
                self._shutdown = error
            self._lock.notify_all()
