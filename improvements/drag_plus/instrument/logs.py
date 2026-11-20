"""JSONL writers that never leave a half-written line behind.

``append_line`` writes one whole line then fsyncs, so a crash loses at most the line in
flight. ``repair_tail`` drops such a partial line on resume. Derived logs are replaced
atomically (write a temp file, fsync, rename), so a reader never sees a torn file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_line(path: Path, record: Any) -> None:
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    created = not path.exists()
    with open(path, "ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _fsync_directory(path.parent)  # make the new directory entry durable too


def repair_tail(path: Path) -> int:
    """Truncate a trailing line that has no newline; return the bytes removed.

    A record and its newline go out in one write, so a missing newline means a torn write.
    """
    if not path.exists():
        return 0
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep == len(data):
        return 0
    with open(path, "r+b") as handle:
        handle.truncate(keep)
        handle.flush()
        os.fsync(handle.fileno())
    return len(data) - keep


def read_lines(path: Path) -> List[dict]:
    """All complete lines of a JSONL file (a missing file is an empty log)."""
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8")
    complete = text[:text.rfind("\n") + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def write_atomic(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_jsonl_atomic(path: Path, records: Iterable[Any]) -> None:
    write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def write_json_atomic(path: Path, value: Any) -> None:
    write_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")
