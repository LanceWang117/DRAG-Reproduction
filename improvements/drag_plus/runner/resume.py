"""Schedule planning and checkpoint loading for ``--resume``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
