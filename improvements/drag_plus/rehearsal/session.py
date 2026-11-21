"""Run a rehearsal: J04b's smoke session policy with resumable per-variant runs.

``run_smoke`` (preflight, server start, model check, stop after a model-side error,
server release in ``finally``) is reused unchanged. Rehearsal differs only in the
per-variant runner: each variant writes to a fixed directory ``<root>/<variant name>``,
so a rerun with the same root resumes unfinished variants and skips finished ones.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from ..runner.local_generator import LocalGeneratorError
from ..runner.local_preflight import Readings
from ..runner.local_server import OllamaServer
from ..runner.local_smoke import SmokeJob, run_smoke
from ..runner.run import RunRequest, run_variant
from ..runner.variant import Variant
from .report import variant_report

DRAG_SECONDS_PER_ROUND = 10  # ~60 s per question at 3 + 3 rounds on llama3.2:3b (J04b)
NAIVE_SECONDS = 10

BackendFactory = Callable[[SmokeJob], Any]


def _complete(out_dir: Path) -> bool:
    manifest = out_dir / "run-manifest.json"
    if not manifest.is_file():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("exit_status") == "complete"


def make_runner(backend_factory: BackendFactory,
                progress: Callable[[str], None] = print) -> Callable[[SmokeJob, Variant], Dict]:
    """A ``run_one`` for ``run_smoke`` that resumes and skips by run directory."""
    def run_one(job: SmokeJob, variant: Variant) -> Dict[str, Any]:
        out_dir = job.output / variant.name
        if _complete(out_dir):
            return variant_report(out_dir)
        request = RunRequest(rows=job.rows, variant=variant, backend=backend_factory(job),
                             out_dir=out_dir, profile=True, interleave=True,
                             resume=(out_dir / "run-manifest.json").is_file(),
                             command=["rehearse_local.py"], progress=progress)
        error = None
        try:
            run_variant(request)
        except LocalGeneratorError as caught:
            error = str(caught)
        report = variant_report(out_dir)
        if error is not None:
            report["problems"].append(error)
            report.update(ok=False, error=error)
        return report
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
