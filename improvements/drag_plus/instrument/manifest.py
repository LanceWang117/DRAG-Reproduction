"""``run-manifest.json``: everything needed to rerun or audit a run.

``collect_environment`` takes its sources as arguments, so tests supply fakes and never
touch git, the GPU or the installed packages.
"""
from __future__ import annotations

import importlib.metadata
import os
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional

PACKAGES = ("vllm", "flashrag-dev", "faiss-gpu", "faiss-cpu", "torch", "transformers")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")


def _git(args: List[str], run: Callable) -> Optional[str]:
    try:
        done = run(["git", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _version(package: str, lookup: Callable[[str], str]) -> Optional[str]:
    try:
        return lookup(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(run: Callable = subprocess.run,
                        environ: Optional[Mapping[str, str]] = None,
                        version_of: Callable[[str], str] = importlib.metadata.version) -> Dict[str, Any]:
    environ = os.environ if environ is None else environ
    status = _git(["status", "--porcelain"], run)
    return {
        "commit": _git(["rev-parse", "HEAD"], run),
        "dirty": bool(status) if status is not None else None,
        "packages": {name: _version(name, version_of) for name in PACKAGES},
        "threads": {name: environ.get(name) for name in THREAD_VARIABLES},
    }


def build_manifest(*, run_id: str, variant: Dict[str, Any], flags: Dict[str, Any],
                   backend: Dict[str, Any], environment: Dict[str, Any], started: str,
                   ended: Optional[str], exit_status: str, resumes: List[dict],
                   command: List[str], arms: List[str], questions: int) -> Dict[str, Any]:
    return {"run_id": run_id, "variant": variant, "flags": flags, "arms": arms,
            "questions": questions, "command": command, "backend": backend,
            "environment": environment, "seed": variant.get("seed"), "started": started,
            "ended": ended, "exit_status": exit_status, "resumes": resumes}
