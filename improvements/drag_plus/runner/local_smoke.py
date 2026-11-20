"""The smoke session: preflight, server, one run per variant, cleanup.

``run_smoke`` is the whole policy. It checks resources before it starts anything, pulls a
small missing model only when the disk allows, runs each variant through the ordinary
variant runner with the local backend, and always lets the server go in a ``finally``
(a server this process did not start is left alone, see ``OllamaServer.close``).
Everything it touches is injected, so the tests use no real server, GPU or model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from .local_backend import build_local_backend
from .local_generator import LocalGeneratorError
from .local_preflight import PreflightError, Readings, check_pull, check_resources
from .local_server import OllamaServer
from .run import RunRequest, default_run_dir, run_variant
from .smoke_summary import summarise_run
from .variant import Variant

MAX_QUESTIONS = 5


@dataclass
class SmokeJob:
    model: str
    rows: List[Dict[str, Any]]
    variants: List[Variant]
    output: Path
    replay: Path
    questions_path: Path
    allow_pull: bool = True


def prepare(server: OllamaServer, model: str, readings: Readings, allow_pull: bool) -> None:
    """Refuse on low resources before starting anything; pull one small model if missing."""
    problems = check_resources(readings)
    if problems:
        raise PreflightError(problems)
    server.ensure_running()
    if server.has_model(model):
        return
    problems = (check_pull(model, readings) if allow_pull
                else [f"model {model!r} is missing and --no-pull was given"])
    if problems:
        raise PreflightError(problems)
    server.pull(model)


def run_variant_once(job: SmokeJob, variant: Variant, base_url: str,
                     progress: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run one variant on the job's questions and summarise its logs.

    A model-side failure (timeouts, a missing model) still leaves a finalised run
    directory; it is summarised and the error is added to the problems.
    """
    backend = build_local_backend(model=job.model, base_url=base_url,
                                  retrieval_path=job.replay, questions_path=job.questions_path)
    out_dir = default_run_dir(job.output, variant.name)
    request = RunRequest(rows=job.rows, variant=variant, backend=backend, out_dir=out_dir,
                         profile=True, command=["smoke_local.py"], progress=progress)
    error = None
    try:
        run_variant(request)
    except LocalGeneratorError as caught:
        error = str(caught)
    result = summarise_run(out_dir)
    if error is not None:
        result["problems"].append(error)
        result.update(ok=False, error=error)
    return result


def run_smoke(job: SmokeJob, server: OllamaServer, readings: Readings,
              run_one: Callable[[SmokeJob, Variant], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Results in variant order; stops after the first variant that hit a model-side error."""
    results: List[Dict[str, Any]] = []
    try:
        prepare(server, job.model, readings, job.allow_pull)
        for variant in job.variants:
            results.append(run_one(job, variant))
            if "error" in results[-1]:
                break
        return results
    finally:
        server.close(job.model)
