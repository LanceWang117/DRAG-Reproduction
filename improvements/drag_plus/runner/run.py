"""Run a variant: schedule, execute with per-question checkpoints, derive the logs.

Fail fast: an exception (or an interrupt) finalises the logs with an error exit status
and is re-raised; the question in flight is not checkpointed, so ``--resume`` retries it.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..decompose.compare import IRCoTPipeline, SelfAskPipeline
from ..instrument import logs, manifest
from ..instrument.timing import CallRecorder
from ..naive import NaivePipeline
from ..pipeline import DragPlusPipeline
from ..retrieval.batch_search import with_batch_search
from ..retrieval.hybrid import config_fingerprint
from . import records, resume
from .backends import Backend, build_retriever
from .variant import Variant

PIPELINES = {"drag": DragPlusPipeline, "naive": NaivePipeline,
             "ircot": IRCoTPipeline, "selfask": SelfAskPipeline}

CHECKPOINTS, TIMINGS, TRANSCRIPTS = "checkpoints.jsonl", "stage-timings.jsonl", "transcripts.jsonl"
MANIFEST, SUMMARY = "run-manifest.json", "summary.json"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def default_run_dir(root: Path, variant: str, stamp: Optional[str] = None) -> Path:
    stamp = stamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{stamp}-{variant}"


@dataclass
class RunRequest:
    rows: List[Dict[str, Any]]
    variant: Variant
    backend: Backend
    out_dir: Path
    profile: bool = False
    interleave: bool = False
    resume: bool = False
    command: Sequence[str] = ()
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], str] = utc_now
    progress: Callable[[str], None] = lambda message: None
    environment: Callable[[], Dict[str, Any]] = manifest.collect_environment
    info: Dict[str, Any] = field(default_factory=dict)


def _flags(request: RunRequest):
    flags = request.variant.flags
    return flags.with_changes(profile_calls=True) if request.profile else flags


class _Pipelines:
    """Build each (arm, dataset) pipeline once; pipelines hold no per-question state."""

    def __init__(self, request: RunRequest):
        self._request, self._flags, self._cache = request, _flags(request), {}
        self._recorder = CallRecorder(request.clock, request.backend.count_tokens)
        self.retriever = self._shared_retriever()

    def _shared_retriever(self):
        """One proxy for every arm, so a query is retrieved once for the whole run.

        The retriever is composed here (dense, or dense + BM25 fused) and only then
        cached; a relative cache path is relative to the run directory, like the logs.
        """
        backend = self._request.backend
        retriever = with_batch_search(build_retriever(self._flags, backend.retriever), self._flags)
        path = self._flags.retrieval_cache
        if path is None:
            return retriever
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self._request.out_dir / resolved
        return backend.cached_retriever(resolved, retriever,
                                        config_fingerprint(self._flags))

    def get(self, arm: str, dataset: str):
        """The pipeline for one (arm, dataset), built once per run."""
        if (arm, dataset) not in self._cache:
            backend = self._request.backend
            cls = PIPELINES[arm]
            extra = {"count_tokens": backend.count_tokens} if cls is DragPlusPipeline else {}
            self._cache[(arm, dataset)] = cls(
                self._flags, backend.generator, self.retriever,
                backend.make_template(arm, dataset), dataset, self._recorder, **extra)
        return self._cache[(arm, dataset)]


def _write_manifest(request: RunRequest, started: str, resumes: List[dict], status: str,
                    ended: Optional[str]) -> None:
    built = manifest.build_manifest(
        run_id=request.out_dir.name, variant=request.variant.raw,
        flags=_flags(request).to_manifest(),
        backend={"name": request.backend.name, **request.backend.describe()},
        environment=request.environment(), started=started, ended=ended,
        exit_status=status, resumes=resumes, command=list(request.command),
        arms=list(request.variant.arms), questions=len(request.rows))
    logs.write_json_atomic(request.out_dir / MANIFEST, built)


def _open_run(request: RunRequest, schedule) -> tuple:
    """Return (progress, started, resumes) after creating or validating the directory."""
    checkpoints = request.out_dir / CHECKPOINTS
    if not request.resume:
        # FlashRAG's Config creates its save_dir inside the run directory, so an empty
        # directory is fine; a directory that already holds a run is not.
        request.out_dir.mkdir(parents=True, exist_ok=True)
        if checkpoints.exists() or (request.out_dir / MANIFEST).exists():
            raise FileExistsError(f"{request.out_dir} already holds a run; use --resume")
        return resume.load_progress(checkpoints, schedule), request.now(), []
    old = resume.check_resumable(request.out_dir, request.variant.raw,
                                 _flags(request).to_manifest(), request.backend.name,
                                 len(request.rows))
    progress = resume.load_progress(checkpoints, schedule)
    entry = {"at": request.now(), "completed_before": len(progress.done),
             "repaired_bytes": progress.repaired_bytes}
    return progress, old.get("started", request.now()), old.get("resumes", []) + [entry]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _batches(schedule, done: Dict[int, Dict[str, Any]], size: int, refill: bool = False):
    """Yield consecutive unfinished slots of one (arm, dataset), at most ``size``.

    With ``refill`` the whole unfinished run is yielded; the pipeline keeps ``size``
    questions live and admits the rest as slots free.
    """
    batch: List[Tuple[int, str, Dict[str, Any]]] = []
    current = None
    for order, (arm, row) in enumerate(schedule):
        if order in done:
            continue
        key = (arm, row["dataset"])
        if batch and key != current:
            yield batch
            batch = []
        current = key
        batch.append((order, arm, row))
        if not refill and len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _execute(request: RunRequest, schedule, progress: resume.Progress) -> bool:
    """Run the unfinished slots in question batches; return True if any question failed."""
    pipelines = _Pipelines(request)
    flags = _flags(request)
    failed = False
    for batch in _batches(schedule, progress.done, flags.question_batch_size,
                          flags.refill_window):
        arm, dataset = batch[0][1], batch[0][2]["dataset"]
        pipeline = pipelines.get(arm, dataset)
        items, meta = [], {}
        for order, arm, row in batch:
            item = records.WorkItem(row)
            items.append((item, (row["dataset"], row["split"], row["id"])))
            meta[id(item)] = (order, arm, row)

        def on_finish(item, state, meta=meta):
            order, arm, row = meta[id(item)]
            record = records.build_record(order, arm, row, state)
            logs.append_line(request.out_dir / CHECKPOINTS, record)
            progress.done[order] = record
            request.progress(f"{arm} {row['dataset']} {order + 1}/{len(schedule)} {row['id']}")

        if isinstance(pipeline, DragPlusPipeline):
            states = pipeline.run_batch(items, batch_size=flags.question_batch_size,
                                        on_finish=on_finish)
        else:
            states = []
            for item, key in items:
                states.append(pipeline.run_item(item, key=key))
                on_finish(item, states[-1])
        failed = failed or any(state.status == "failed" for state in states)
    return failed


def _write_logs(request: RunRequest, done: Dict[int, Dict[str, Any]]) -> None:
    timings, transcripts = records.derive_logs(
        list(done.values()), request.out_dir.name, request.variant.name)
    logs.write_jsonl_atomic(request.out_dir / TRANSCRIPTS, transcripts)
    if _flags(request).profile_calls:
        logs.write_jsonl_atomic(request.out_dir / TIMINGS, timings)


def _summarise(request: RunRequest, done: Dict[int, Dict[str, Any]]) -> None:
    groups: Dict[tuple, List[dict]] = {}
    for record in sorted(done.values(), key=lambda r: r["order"]):
        groups.setdefault((record["arm"], record["dataset"]), []).append(record)
    results = [{"arm": arm, "dataset": dataset, "count": len(group),
                "metrics": request.backend.evaluate(arm, dataset, group)}
               for (arm, dataset), group in groups.items()]
    logs.write_json_atomic(request.out_dir / SUMMARY, {"results": results})


def run_variant(request: RunRequest) -> Dict[str, Any]:
    """Run (or resume) the variant; returns the final manifest."""
    schedule = resume.plan(request.rows, request.variant.arms, request.interleave)
    progress, started, resumes = _open_run(request, schedule)
    _write_manifest(request, started, resumes, "running", None)
    status = "complete"
    try:
        failed = _execute(request, schedule, progress)
        _write_logs(request, progress.done)
        _summarise(request, progress.done)
        if failed:
            status = "failed"
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        with contextlib.suppress(Exception):
            _write_logs(request, progress.done)
        raise
    finally:
        _write_manifest(request, started, resumes, status, request.now())
    return read_json(request.out_dir / MANIFEST)
