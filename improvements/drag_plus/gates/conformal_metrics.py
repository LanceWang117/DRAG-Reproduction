"""Coverage and selective-risk metrics for the J21 conformal evaluation.

``clopper_pearson`` is the exact two-sided binomial interval (Clopper & Pearson, 1934),
computed by bisection on the binomial CDF in log space: the gate has no SciPy and must
not grow one. ``summarise`` turns cross-fitted rows into coverage, set size, abstention,
selective accuracy and the accuracy at 50/70/90% coverage targets. The SVG writer is
dependency-free, matching the J20 scripts, because ``requirements-ci.txt`` has no
Matplotlib.

Rows are dictionaries with ``dataset``, ``covered``, ``set_size``, ``abstained``,
``answered_correct``, ``argmax_correct``, ``confidence`` and ``fold``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

TARGETS = (0.5, 0.7, 0.9)


def _log_pmf(count: int, total: int, probability: float) -> float:
    return (math.lgamma(total + 1) - math.lgamma(count + 1)
            - math.lgamma(total - count + 1) + count * math.log(probability)
            + (total - count) * math.log1p(-probability))


def _binom_cdf(successes: int, total: int, probability: float) -> float:
    """P(X <= successes) for X ~ Binomial(total, probability), summed in log space."""
    if successes < 0:
        return 0.0
    if successes >= total or probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    logs = [_log_pmf(i, total, probability) for i in range(successes + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(value - top) for value in logs))


def _solve(cdf, target: float) -> float:
    """Bisect the decreasing binomial CDF to the probability where it equals target."""
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if cdf(middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(successes: int, total: int,
                    confidence: float = 0.95) -> Tuple[float, float]:
    """Exact two-sided binomial interval (Clopper & Pearson, 1934)."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("clopper_pearson needs 0 <= successes <= total and total > 0")
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else \
        _solve(lambda p: _binom_cdf(successes - 1, total, p), 1.0 - tail)
    high = 1.0 if successes == total else \
        _solve(lambda p: _binom_cdf(successes, total, p), tail)
    return low, high


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(rows: Sequence[dict],
              thresholds: Mapping[Tuple[int, float], float]) -> dict:
    """Coverage, set size, abstention and selective accuracy from held-out rows."""
    total, covered = len(rows), sum(row["covered"] for row in rows)
    answered = [row for row in rows if not row["abstained"]]
    low, high = clopper_pearson(covered, total)
    targets = {}
    for target in TARGETS:
        selected = [row for row in rows
                    if row["confidence"] >= thresholds[(row["fold"], target)]]
        targets[f"{target:.2f}"] = {
            "coverage": len(selected) / total,
            "selective_accuracy": _mean([row["argmax_correct"] for row in selected]),
        }
    per_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        per_dataset[dataset] = {
            "n": len(subset), "coverage": _mean([row["covered"] for row in subset]),
            "abstention_rate": _mean([row["abstained"] for row in subset])}
    return {"n": total, "coverage": covered / total, "cp_low": low, "cp_high": high,
            "mean_set_size": _mean([row["set_size"] for row in rows]),
            "abstention_rate": _mean([row["abstained"] for row in rows]),
            "full_coverage_accuracy": _mean([row["argmax_correct"] for row in rows]),
            "selective_accuracy": _mean([row["answered_correct"] for row in answered]),
            "selective_coverage": len(answered) / total,
            "selective_at": targets, "per_dataset": per_dataset}


def curve(rows: Sequence[dict], points: int = 40) -> List[Tuple[float, float]]:
    """Accuracy after answering the ``k`` most confident rows, for a sampled ``k``."""
    ordered = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    result, correct, step = [(0.0, 0.0)], 0, max(1, len(ordered) // points)
    for index, row in enumerate(ordered, 1):
        correct += int(row["argmax_correct"])
        if index % step == 0 or index == len(ordered):
            result.append((index / len(ordered), correct / index))
    return result


def _xy(coverage: float, accuracy: float, width: int, height: int) -> str:
    pad = 30
    x = pad + coverage * (width - 2 * pad)
    y = height - pad - accuracy * (height - 2 * pad)
    return f"{x:.1f},{y:.1f}"


def risk_coverage_svg(rows: Sequence[dict], path: Any) -> None:
    """Write a small dependency-free SVG risk-coverage curve."""
    width, height = 360, 240
    line = " ".join(_xy(coverage, accuracy, width, height)
                    for coverage, accuracy in curve(rows))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'<line x1="30" y1="210" x2="330" y2="210" stroke="black"/>'
           f'<line x1="30" y1="30" x2="30" y2="210" stroke="black"/>'
           f'<polyline points="{line}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
           f'<text x="150" y="234" font-size="11">coverage</text>'
           f'<text x="4" y="120" font-size="11" transform="rotate(-90 12,120)">accuracy</text>'
           f'</svg>\n')
    Path(path).write_text(svg, encoding="utf-8")
