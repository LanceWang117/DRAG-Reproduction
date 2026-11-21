"""Run a rehearsal: J04b's smoke session policy with resumable per-variant runs.

``run_smoke`` (preflight, server start, model check, stop after a model-side error,
server release in ``finally``) is reused unchanged. Rehearsal differs only in the
per-variant runner. Each variant writes to a fixed directory ``<root>/runs/<name>``, so
a rerun with the same root resumes unfinished variants. A finished variant is reused
untouched only after its manifest is checked against the current settings (variant,
flags, question count, backend and model). A mismatch fails that arm alone, so the other
arms still report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from ..runner import resume
from ..runner.local_generator import LocalGeneratorError
from ..runner.local_preflight import Readings
from ..runner.local_server import OllamaServer
from ..runner.local_smoke import SmokeJob, run_smoke
from ..runner.run import RunRequest, run_variant
from ..runner.variant import Variant
from .report import variant_report

# Measured in the L01 live rehearsal (llama3.2:3b, GTX 1660 Ti, 6 dev questions): baseline DRAG
# 174 s/question over 6 rounds, the 1 + 1 round variant 46 s/question, Naive 5 s/question.
DRAG_SECONDS_PER_ROUND = 28
NAIVE_SECONDS = 5

BackendFactory = Callable[[SmokeJob], Any]


def run_dir(root: Path, variant: Variant) -> Path:
    return Path(root) / "runs" / variant.name


def _manifest(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "run-manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _check_finished(out_dir: Path, job: SmokeJob, variant: Variant, backend: Any) -> None:
    """Raise ResumeError unless the finished run used exactly these settings."""
    flags = variant.flags.with_changes(profile_calls=True).to_manifest()
    old = resume.check_resumable(out_dir, variant.raw, flags, backend.name, len(job.rows))
    model, current = old.get("backend", {}).get("model"), backend.describe().get("model")
    if model != current:
        raise resume.ResumeError(f"cannot reuse {out_dir}: it ran {model!r}, not {current!r}")


def _failed(out_dir: Path, variant: Variant, reason: str, expected: int) -> Dict[str, Any]:
    return {"variant": variant.name, "run_dir": str(out_dir), "ok": False, "completed": 0,
            "expected": expected, "parse_failures": 0, "empty_outputs": 0, "schema_errors": 0,
            "problems": [reason]}


def _execute(request: RunRequest) -> Dict[str, Any]:
    error = None
    try:
        run_variant(request)
    except LocalGeneratorError as caught:
        error = str(caught)
    report = variant_report(request.out_dir)
    if error is not None:
        report["problems"].append(error)
        report.update(ok=False, error=error)
    return report


def make_runner(backend_factory: BackendFactory,
                progress: Callable[[str], None] = print) -> Callable[[SmokeJob, Variant], Dict]:
    """A ``run_one`` for ``run_smoke`` that resumes, reuses finished runs, and isolates arms."""
    def run_one(job: SmokeJob, variant: Variant) -> Dict[str, Any]:
        out_dir = run_dir(job.output, variant)
        expected = len(job.rows) * len(variant.arms)
        backend = backend_factory(job)
        old = _manifest(out_dir)
        try:
            if old.get("exit_status") == "complete":
                _check_finished(out_dir, job, variant, backend)
                return variant_report(out_dir)
            request = RunRequest(rows=job.rows, variant=variant, backend=backend,
                                 out_dir=out_dir, profile=True, interleave=True,
                                 resume=bool(old), command=["rehearse_local.py"],
                                 progress=progress)
            return _execute(request)
        except resume.ResumeError as error:
            return _failed(out_dir, variant, str(error), expected)
    return run_one


def run_rehearsal(job: SmokeJob, server: OllamaServer, readings: Readings,
                  backend_factory: BackendFactory,
                  progress: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    """Reports in variant order (see ``run_smoke`` for the stopping rule)."""
    return run_smoke(job, server, readings, make_runner(backend_factory, progress))


def estimate_minutes(variants: Sequence[Variant], questions: int) -> float:
    """A planning estimate for llama3.2:3b on the laptop; never a measurement."""
    seconds = 0.0
    for variant in variants:
        flags = variant.flags
        if "drag" in variant.arms:
            rounds = flags.max_query_debate_rounds + flags.max_answer_debate_rounds
            seconds += questions * DRAG_SECONDS_PER_ROUND * max(rounds, 1)
        if "naive" in variant.arms:
            seconds += questions * NAIVE_SECONDS
    return seconds / 60
