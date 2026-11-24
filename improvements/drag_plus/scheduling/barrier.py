"""Cross-question barrier scheduler.

A stage call that needs the generator or retriever submits a ``Request`` and blocks.
One worker thread runs each question, so a step is a pure function of that question's
state; the scheduler itself holds no question data. At every barrier it waits until all
live workers are blocked, groups their requests by ``(kind, stage, round, role, params,
labels)``, executes exactly one backend call per group, and hands each request its own
slice of the result. All search requests of a step are one group (J16), so every ready
query goes to one ``batch_search``; the proxy applies the query cap. Finished questions
leave the barrier; the rest continue.

With a ``RefillWindow`` (J16) a finished question's slot is refilled at once.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .refill import (BarrierContractError, BarrierError, GENERATE, SCORES, SEARCH,
                     Request, SchedulerStats)


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
        self._threads: List[threading.Thread] = []
        self._refill: Optional[Any] = None
        self._shutdown: Optional[BaseException] = None
        self.stats = SchedulerStats()

    def run(self, workers: Sequence[Tuple[Any, Callable[[], None]]],
            on_done: Optional[Callable[[Any], None]] = None,
            refill: Optional[Any] = None) -> None:
        """Run ``(state, target)`` pairs, calling ``on_done(state)`` when each finishes.

        With ``refill`` (a :class:`RefillWindow`), ``workers`` is ignored because the
        window already holds every target; every finished slot is refilled from its
        pending queue until it is exhausted.
        """
        self._refill = refill
        self._workers, self._threads = [], []
        try:
            if refill is None:
                for state, target in workers:
                    self._spawn(state, target)
            else:
                self._admit()
            self._loop(on_done)
        except _Aborted:
            pass
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            for thread in self._threads:
                thread.join(timeout=5)
        if self._shutdown is not None:
            raise self._shutdown

    def _spawn(self, state: Any, target: Callable[[], None]) -> None:
        """Start one worker thread for a question; call under the lock when running."""
        worker = _Worker(state, target, len(self._workers))
        self._workers.append(worker)
        thread = threading.Thread(target=self._worker_main, args=(worker,), daemon=True)
        self._threads.append(thread)
        thread.start()

    def _admit(self) -> None:
        """Fill the window's free slots from its pending queue."""
        with self._lock:
            live = sum(1 for worker in self._workers if not worker.done)
            for state, target in self._refill.fill(live):
                self._spawn(state, target)
            self._lock.notify_all()

    def _drained(self) -> bool:
        """True when every worker finished, was announced, and no pending question remains."""
        if not all(worker.done for worker in self._workers):
            return False
        if any(worker.done and not worker.notified for worker in self._workers):
            return False
        return self._refill is None or self._refill.exhausted

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
            if self._refill is not None:
                self._admit()
            if batch is not None:
                self._dispatch(batch)
            elif self._shutdown is not None:
                raise _Aborted
            elif self._drained():
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
        groups = self._groups(requests)
        self.stats.barriers += 1
        if self._refill is not None:
            self._refill.record(len(requests))
        for group in groups:
            self._run_group(group)
        with self._lock:
            for request in requests:
                request.done = True
            self._lock.notify_all()

    @staticmethod
    def _groups(requests: List[Request]) -> List[List[Request]]:
        """Group one step: all searches together, other kinds by request key.

        A step may mix query and answer stages or rounds; search requests have no
        generation params or labels, so one ``batch_search`` can serve every ready
        query. The proxy splits that group at ``retrieval_batch_size``.
        """
        searches = [request for request in requests if request.kind == SEARCH]
        groups: Dict[tuple, List[Request]] = {}
        for request in requests:
            if request.kind != SEARCH:
                groups.setdefault(request.key(), []).append(request)
        if searches:
            groups[("search",)] = searches
        return list(groups.values())

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
