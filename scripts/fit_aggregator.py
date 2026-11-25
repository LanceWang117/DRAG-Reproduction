"""Fit the J19 weighted-aggregator coefficients from labelled candidate rows.

Input is JSONL: one object per candidate with ``features`` = {conf, s_faith, agr} and
``target`` in [0, 1] (1 = the candidate is correct). The script solves a non-negative
least-squares problem without numpy and writes the weights JSON consumed by
``drag_plus.grounding.scorer.load_weights``: the three weights, the training split and
the SHA-256 of the input file. A ``test`` split is refused, as are malformed rows, a
singular design and an all-zero fit; the script exits non-zero rather than writing a
silent fallback model.

Edge cases: empty input, missing/extra feature names, non-finite values, targets
outside [0, 1], an unwritable output and a singular design are errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from drag_plus.grounding.scorer import SIGNAL_NAMES, AggregatorWeights

Row = Dict[str, float]


class FitError(ValueError):
    """The input rows or the fit itself cannot produce a valid weights file."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FitError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise FitError(f"{label} must be finite")
    return number


def _row(data: Any, line: int) -> Row:
    if not isinstance(data, dict):
        raise FitError(f"line {line}: row must be a JSON object")
    features = data.get("features")
    if not isinstance(features, dict) or set(features) != set(SIGNAL_NAMES):
        raise FitError(f"line {line}: features must be exactly {list(SIGNAL_NAMES)}")
    row: Row = {}
    for name in SIGNAL_NAMES:
        value = _number(features[name], f"line {line}: feature {name!r}")
        if not 0.0 <= value <= 1.0:
            raise FitError(f"line {line}: feature {name!r} must be in [0, 1]")
        row[name] = value
    target = _number(data.get("target"), f"line {line}: target")
    if not 0.0 <= target <= 1.0:
        raise FitError(f"line {line}: target must be in [0, 1]")
    row["target"] = target
    return row


def read_rows(path: Any) -> List[Row]:
    """Parse one JSON object per non-blank line; every row is validated."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise FitError(f"cannot read {path!r}: {exc}") from exc
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FitError(f"line {number}: invalid JSON: {exc}") from exc
        rows.append(_row(data, number))
    if not rows:
        raise FitError("input has no candidate rows")
    return rows


def _normal_equations(rows: Sequence[Row], active: Sequence[int], l2: float):
    size = len(active)
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    for row in rows:
        values = [row[SIGNAL_NAMES[index]] for index in active]
        for i, left in enumerate(values):
            vector[i] += left * row["target"]
            for j, right in enumerate(values):
                matrix[i][j] += left * right
    for index in range(size):
        matrix[index][index] += l2
    return matrix, vector


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]):
    """Gaussian elimination with partial pivoting; ``None`` when singular."""
    size = len(vector)
    work = [list(matrix[index]) + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        for index in range(column, size + 1):
            work[column][index] /= scale
        for row in range(size):
            factor = work[row][column]
            if row != column and factor != 0.0:
                for index in range(column, size + 1):
                    work[row][index] -= factor * work[column][index]
    return [work[index][size] for index in range(size)]


def fit_weights(rows: Sequence[Row], l2: float = 0.0) -> Dict[str, float]:
    """Non-negative least squares over the three signals; at least one is positive."""
    active = list(range(len(SIGNAL_NAMES)))
    while active:
        matrix, vector = _normal_equations(rows, active, l2)
        solution = _solve(matrix, vector)
        if solution is None:
            raise FitError("design matrix is singular")
        if all(value >= -1e-9 for value in solution):
            weights = {name: 0.0 for name in SIGNAL_NAMES}
            for index, value in zip(active, solution):
                weights[SIGNAL_NAMES[index]] = max(0.0, value)
            if sum(weights.values()) <= 0.0:
                raise FitError("fit is all zero")
            return weights
        worst = min(range(len(solution)), key=lambda index: solution[index])
        active.pop(worst)
    raise FitError("no non-negative fit")


def write_weights(path: Any, model: AggregatorWeights) -> None:
    text = json.dumps(model.as_dict(), indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL candidate rows")
    parser.add_argument("--out", required=True, help="weights JSON to write")
    parser.add_argument("--split", default="train", help="training split name; not 'test'")
    parser.add_argument("--l2", type=float, default=1e-8, help="ridge penalty")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.l2) or args.l2 < 0.0:
            raise FitError("--l2 must be finite and >= 0")
        if args.split.strip().casefold() == "test":
            raise FitError("refusing to fit on the test split")
        rows = read_rows(args.input)
        weights = fit_weights(rows, l2=args.l2)
        digest = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
        model = AggregatorWeights(training_split=args.split,
                                  manifest_sha256=digest, **weights)
        write_weights(args.out, model)
    except (FitError, OSError, ValueError) as exc:
        print(f"fit_aggregator: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}: {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
