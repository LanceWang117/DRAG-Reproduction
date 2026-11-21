"""Refuse a FAISS-GPU configuration that cannot fit beside vLLM's reservation.

The A100 target has one 80 GiB card. The model is deliberately simple and
conservative::

    required = utilization * total
             + index_bytes / 2        # FAISS clones the index in fp16
             + 0.5 GiB                # the E5 query encoder
             + 3 GiB                  # allocator/CUDA context margin

``nvidia-smi`` is only called on the server path; every computation here is a pure
function and the tests inject the GPU reading. A missing card, a non-numeric reading
or an out-of-range ratio is an error, never a silent pass.

Usage (GPU host)::

    python scripts/preflight_gpu.py --index-bytes "$(stat -c%s wiki_corpus/e5_flat_inner.index)" \
        --gpu-memory-utilization 0.45

    python scripts/preflight_gpu.py --index-gib 64.5 --gpu-total-gib 80 \
        --gpu-memory-utilization 0.45          # offline, no nvidia-smi
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys

GIB = 1024 ** 3
MIB_PER_GIB = 1024.0
E5_GIB = 0.5
MARGIN_GIB = 3.0
SMI_MEMORY_QUERY = ["nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits"]


class PreflightError(RuntimeError):
    """The configuration does not fit, or the GPU reading is unusable."""


def _number(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Return a finite float, rejecting NaN/inf and (by default) non-positive values."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}, got {value!r}")
    return float(value)


def required_gib(total_gib: float, gpu_memory_utilization: float, index_gib: float,
                 e5_gib: float = E5_GIB, margin_gib: float = MARGIN_GIB) -> float:
    """Memory the run needs, in GiB; raises ValueError on impossible inputs."""
    total = _number("total_gib", total_gib)
    index = _number("index_gib", index_gib, allow_zero=True)
    e5 = _number("e5_gib", e5_gib, allow_zero=True)
    margin = _number("margin_gib", margin_gib, allow_zero=True)
    utilization = _number("gpu_memory_utilization", gpu_memory_utilization)
    if utilization > 1.0:
        raise ValueError(f"gpu_memory_utilization must be <= 1, got {utilization!r}")
    return utilization * total + index / 2.0 + e5 + margin


def check_fit(total_gib: float, gpu_memory_utilization: float, index_gib: float,
              e5_gib: float = E5_GIB, margin_gib: float = MARGIN_GIB) -> dict:
    """Return the breakdown, or raise PreflightError with a message naming the fix."""
    required = required_gib(total_gib, gpu_memory_utilization, index_gib, e5_gib, margin_gib)
    breakdown = {
        "total_gib": float(total_gib),
        "required_gib": required,
        "vllm_gib": gpu_memory_utilization * float(total_gib),
        "index_fp16_gib": float(index_gib) / 2.0,
        "e5_gib": float(e5_gib),
        "margin_gib": float(margin_gib),
    }
    if required > float(total_gib):
        raise PreflightError(
            f"Insufficient GPU memory: need {required:.2f} GiB of {total_gib:.2f} GiB "
            f"(vLLM {breakdown['vllm_gib']:.2f} + fp16 index "
            f"{breakdown['index_fp16_gib']:.2f} + E5 {e5_gib:.2f} + margin "
            f"{margin_gib:.2f}). Lower gpu_memory_utilization (now "
            f"{gpu_memory_utilization}) or use a smaller index.")
    return breakdown


def check_fit_bytes(total_gib: float, gpu_memory_utilization: float, index_bytes: int,
                    e5_gib: float = E5_GIB, margin_gib: float = MARGIN_GIB) -> dict:
    """``check_fit`` for an index size measured in bytes."""
    size = _number("index_bytes", index_bytes, allow_zero=True)
    return check_fit(total_gib, gpu_memory_utilization, size / GIB, e5_gib, margin_gib)


def parse_smi_memory(text: str) -> float:
    """Sum ``nvidia-smi --query-gpu=memory.total`` lines (MiB) into GiB."""
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError as error:
            raise PreflightError(f"unexpected nvidia-smi memory line: {line!r}") from error
    if not values:
        raise PreflightError("nvidia-smi reported no GPU memory rows")
    if any(value <= 0 for value in values):
        raise PreflightError(f"nvidia-smi reported a non-positive memory value: {text!r}")
    return sum(values) / MIB_PER_GIB


def read_gpu_total_gib(runner=subprocess.run) -> float:
    """Read the visible GPUs' total memory; inject ``runner`` in tests, never here."""
    try:
        done = runner(SMI_MEMORY_QUERY, capture_output=True, text=True, check=False,
                      timeout=15)
    except OSError as error:
        raise PreflightError(
            "nvidia-smi could not be run; this check needs a GPU host") from error
    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "no stderr"
        raise PreflightError(f"nvidia-smi failed (exit {done.returncode}): {tail}")
    return parse_smi_memory(done.stdout)


def _index_gib(args) -> float:
    if args.index_bytes is not None:
        return args.index_bytes / GIB
    return args.index_gib


def main(argv=None) -> int:
    """Check the given size and card, print the result and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--index-bytes", type=int, help="Size of the fp32 index file")
    size.add_argument("--index-gib", type=float, help="Index size in GiB (offline checks)")
    parser.add_argument("--gpu-memory-utilization", type=float, required=True,
                        help="vLLM's gpu_memory_utilization (0, 1]")
    parser.add_argument("--gpu-total-gib", type=float,
                        help="Skip nvidia-smi and use this total (offline checks)")
    parser.add_argument("--e5-gib", type=float, default=E5_GIB)
    parser.add_argument("--margin-gib", type=float, default=MARGIN_GIB)
    args = parser.parse_args(argv)
    try:
        total = args.gpu_total_gib if args.gpu_total_gib is not None \
            else read_gpu_total_gib()
        result = check_fit(total, args.gpu_memory_utilization, _index_gib(args),
                           args.e5_gib, args.margin_gib)
    except (PreflightError, ValueError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        return 1
    print(f"Preflight passed: need {result['required_gib']:.2f} GiB of "
          f"{result['total_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
