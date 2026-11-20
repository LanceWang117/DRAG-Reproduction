"""Schedule planning and checkpoint loading for ``--resume``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..instrument import logs

Slot = Tuple[str, Dict[str, Any]]


class ResumeError(ValueError):
    """The run directory does not match the schedule being resumed."""


def plan(rows: Sequence[Dict[str, Any]], arms: Sequence[str], interleave: bool) -> List[Slot]:
    """Arms in the order given; ``interleave`` alternates arms question by question."""
    if interleave:
        return [(arm, row) for row in rows for arm in arms]
    return [(arm, row) for arm in arms for row in rows]


def identity(arm: str, row: Dict[str, Any]) -> Tuple[str, str, str, Any]:
    return (arm, row["dataset"], row["split"], row["id"])


@dataclass
class Progress:
    done: Dict[int, Dict[str, Any]]
    repaired_bytes: int


def load_progress(path: Path, schedule: List[Slot]) -> Progress:
    """Drop a torn last line, then check every record against its schedule slot."""
    repaired = logs.repair_tail(path)
    done: Dict[int, Dict[str, Any]] = {}
    for record in logs.read_lines(path):
        order = record["order"]
        if order >= len(schedule) or identity(*schedule[order]) != (
                record["arm"], record["dataset"], record["split"], record["id"]):
            raise ResumeError(
                f"checkpoint {order} does not match the schedule; resume with the same "
                f"variant, questions and --interleave setting")
        done[order] = record
    return Progress(done, repaired)


def check_resumable(out_dir: Path, variant: Dict[str, Any], flags: Dict[str, Any],
                    backend_name: str, questions: int) -> Dict[str, Any]:
    """Refuse to resume under different settings; return the earlier manifest ({} if none).

    The effective flags include ``profile_calls``, so resuming without ``--profile`` after
    a profiled start cannot leave a half-profiled timing log.
    """
    if not out_dir.is_dir():
        raise ResumeError(f"cannot resume: {out_dir} does not exist")
    path = out_dir / "run-manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    expected = {"variant": variant, "flags": flags, "questions": questions}
    for key, value in expected.items():
        if old and old.get(key) != value:
            raise ResumeError(f"cannot resume: '{key}' differs from the original run "
                              f"({old.get(key)!r} vs {value!r})")
    if old and old.get("backend", {}).get("name") != backend_name:
        raise ResumeError(f"cannot resume: the original run used another backend "
                          f"({old.get('backend', {}).get('name')!r} vs {backend_name!r})")
    return old


def previous_manifest(out_dir: Path) -> Optional[Dict[str, Any]]:
    path = out_dir / "run-manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
