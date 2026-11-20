"""Preflight for the local smoke run: resource floors checked on injected readings.

Only ``read_hardware`` touches the machine, and it takes its sources as arguments, so no
test ever reads the real GPU (the PC's GTX 1660 Ti would report the wrong card for an
A100 job) or the real memory. Everything else works on a ``Readings`` value.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

MIN_VRAM_GIB = 3.0
MIN_RAM_GIB = 4.0
MIN_DISK_GIB = 1.0
MIN_PULL_DISK_GIB = 10.0
MAX_PULL_BILLIONS = 4.0
_GIB = 1024 ** 3
_SIZE_TAG = re.compile(r"(\d+(?:\.\d+)?)b")


class PreflightError(RuntimeError):
    """The smoke run must not start; ``problems`` says why, one line each."""

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Readings:
    vram_gib: Optional[float]  # free VRAM; None when no NVIDIA GPU answered
    ram_gib: float             # available RAM
    disk_gib: float            # free disk where the output and the model go


def parse_meminfo(text: str) -> float:
    """Available memory in GiB from the text of ``/proc/meminfo``."""
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.MULTILINE)
    if match is None:
        raise ValueError("no MemAvailable line in the memory report")
    return int(match.group(1)) * 1024 / _GIB


def parse_nvidia_smi(output: str) -> Optional[float]:
    """Free VRAM of the first GPU in GiB from ``memory.free`` in MiB, one line per GPU."""
    for line in output.splitlines():
        if line.strip().isdigit():
            return int(line.strip()) / 1024
    return None


def read_hardware(disk_path: Path, *, run: Callable[..., Any] = subprocess.run,
                  meminfo: Path = Path("/proc/meminfo"),
                  disk_usage: Callable[[Any], Any] = shutil.disk_usage) -> Readings:
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    try:
        done = run(command, capture_output=True, text=True, timeout=15)
        vram = parse_nvidia_smi(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        vram = None
    ram = parse_meminfo(meminfo.read_text(encoding="utf-8"))
    return Readings(vram, ram, disk_usage(disk_path).free / _GIB)


def check_resources(readings: Readings) -> List[str]:
    """Problems that stop any smoke run: too little VRAM, RAM or output disk."""
    problems = []
    if readings.vram_gib is None:
        problems.append("no NVIDIA GPU reading (nvidia-smi); the smoke run needs "
                        f"{MIN_VRAM_GIB:g} GiB of free VRAM")
    elif readings.vram_gib < MIN_VRAM_GIB:
        problems.append(f"free VRAM is {readings.vram_gib:.1f} GiB, need {MIN_VRAM_GIB:g} GiB")
    if readings.ram_gib < MIN_RAM_GIB:
        problems.append(f"available RAM is {readings.ram_gib:.1f} GiB, need {MIN_RAM_GIB:g} GiB")
    if readings.disk_gib < MIN_DISK_GIB:
        problems.append(f"free disk is {readings.disk_gib:.1f} GiB, need {MIN_DISK_GIB:g} GiB")
    return problems


def check_pull(model: str, readings: Readings) -> List[str]:
    """Problems that stop an automatic pull: an unknown or large model, or too little disk."""
    problems = []
    size = _SIZE_TAG.match(model.rpartition(":")[2].lower()) if ":" in model else None
    if size is None or float(size.group(1)) > MAX_PULL_BILLIONS:
        problems.append(f"{model!r} is missing and is not a tag of {MAX_PULL_BILLIONS:g}B "
                        f"parameters or fewer, so it is never pulled automatically; "
                        f"run `ollama pull {model}` yourself")
    if readings.disk_gib < MIN_PULL_DISK_GIB:
        problems.append(f"free disk is {readings.disk_gib:.1f} GiB, need "
                        f"{MIN_PULL_DISK_GIB:g} GiB to pull a model")
    return problems
