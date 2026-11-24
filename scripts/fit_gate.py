"""Fit, cross-validate and calibrate the J14 escalation gate from logged transcripts.

Input is JSONL: one record per answered question. A record may carry an explicit
``features`` object plus ``correct``/``wrong``, or the fields needed to recompute the
four label-free features: ``agent_answers`` (or ``pred``/``raw_pred``), ``query_pool``
(or ``passages``), ``judge_score`` and ``strategy_qa``. Gold labels are used only for
the target and are never visible to the feature code.

The script writes a small model JSON (feature names, coefficients, intercept, Platt
calibration map, training split, manifest hash) and a dependency-free reliability
SVG. It refuses the test split and exits non-zero when cross-validated AUROC is below
``--min-auroc``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drag_plus.gates.features import FEATURE_NAMES, compute_features

_WORD = re.compile(r"[a-z0-9]+")


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _fit_logistic(x, y, l2=1.0, steps=800, lr=0.5):
    """Small L2 logistic regression by full-batch gradient descent; no sklearn."""
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(steps):
        probabilities = _sigmoid(x @ weights + bias)
        error = probabilities - y
        weights -= lr * (x.T @ error / len(y) + l2 * weights / len(y))
        bias -= lr * float(np.mean(error))
    return weights, float(bias)


def _fit_platt(logits, y, steps=500, lr=0.2):
    """Fit ``sigmoid(a * logit + b)`` on out-of-fold predictions (Platt scaling)."""
    a, b = 1.0, 0.0
    for _ in range(steps):
        probabilities = _sigmoid(a * logits + b)
        error = probabilities - y
        a -= lr * float(np.mean(error * logits))
        b -= lr * float(np.mean(error))
    return float(a), float(b)


def _ties_average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def auroc(scores, labels) -> float:
    """Rank AUROC for ``wrong == 1``; ties share their average rank."""
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = _ties_average_ranks(np.asarray(scores, dtype=float))
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def cross_validate(x, y, folds, seed, l2):
    """Out-of-fold raw logits; every training fold must contain both classes."""
    count = len(y)
    folds = max(2, min(int(folds), count))
    order = np.random.default_rng(seed).permutation(count)
    out_of_fold = np.zeros(count)
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        if len(set(y[train])) < 2:
            raise ValueError("a cross-validation training fold has only one class")
        weights, bias = _fit_logistic(x[train], y[train], l2)
        out_of_fold[test] = x[test] @ weights + bias
    return out_of_fold, folds


def _pool(record):
    """Passages as a query pool; a bare list is one query."""
    pool = record.get("query_pool")
    if isinstance(pool, dict):
        return {str(query): list(passages) for query, passages in pool.items()}
    passages = record.get("passages") or []
    return {"q": [item if isinstance(item, dict) else {"contents": str(item)}
                   for item in passages]}


def _state(record):
    """A read-only state-shaped object for ``compute_features``."""
    answers = record.get("agent_answers")
    if not answers:
        answers = [record.get("pred") or record.get("raw_pred") or ""]
    messages = {f"Agent {index}": [{"content": str(answer)}]
                for index, answer in enumerate(answers)}
    return SimpleNamespace(query_pool=_pool(record), answer_messages=messages,
                           judge_score=record.get("judge_score"))


def features_from_record(record):
    """Four finite features, from an explicit object or recomputed from the transcript."""
    if "features" in record:
        source = record["features"]
        if not isinstance(source, dict) or set(source) != set(FEATURE_NAMES):
            raise ValueError("features must be an object with exactly the four J14 names")
        values = {name: float(source[name]) for name in FEATURE_NAMES}
    else:
        values = dict(compute_features(_state(record), bool(record.get("strategy_qa"))))
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("every feature must be finite")
    return values
def _wrong(record):
    if "wrong" in record:
        return bool(record["wrong"])
    if "correct" in record:
        return not bool(record["correct"])
    pred = str(record.get("pred") or record.get("raw_pred") or "")
    gold = record.get("gold_answers", record.get("gold"))
    golds = gold if isinstance(gold, list) else [gold]
    if not pred or not golds or any(item is None for item in golds):
        raise ValueError("a record needs correct/wrong or pred plus gold_answers/gold")
    normal = lambda text: " ".join(_WORD.findall(str(text).lower()))
    return normal(pred) not in {normal(item) for item in golds}


def read_records(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"line {number}: expected a JSON object")
        records.append(record)
    if not records:
        raise ValueError("input has no records")
    return records


def _svg_xy(value, width, height, pad):
    x = pad + (width - 2 * pad) * value[0]
    y = height - pad - (height - 2 * pad) * value[1]
    return f"{x:.1f},{y:.1f}"


def write_reliability(path, probabilities, labels, bins=5):
    """Write a dependency-free reliability diagram (predicted vs observed wrong rate)."""
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    points = []
    for index in range(bins):
        upper = edges[index + 1]
        mask = ((probabilities >= edges[index]) & (probabilities < upper)
                if index < bins - 1 else
                ((probabilities >= edges[index]) & (probabilities <= upper)))
        if mask.any():
            points.append((float(probabilities[mask].mean()), float(labels[mask].mean())))
    width = height = 320
    pad = 40
    diagonal = " ".join(_svg_xy((v, v), width, height, pad) for v in (0.0, 1.0))
    curve = " ".join(_svg_xy(point, width, height, pad) for point in points)
    circles = "".join(f'<circle cx="{_svg_xy(point, width, height, pad).split(",")[0]}" '
                      f'cy="{_svg_xy(point, width, height, pad).split(",")[1]}" r="4"/>'
                      for point in points)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'<text x="{pad}" y="22" font-family="sans-serif" font-size="13">'
           f'Reliability ({len(points)} populated bins)</text>'
           f'<polyline points="{diagonal}" fill="none" stroke="#999" stroke-dasharray="4"/>'
           f'<polyline points="{curve}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
           f'{circles}</svg>\n')
    Path(path).write_text(svg, encoding="utf-8")


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--min-auroc", type=float, default=0.70)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.split == "test":
        print("fit_gate: refusing to fit on the test split")
        return 2
    if args.folds < 2:
        print("fit_gate: --folds must be at least 2"); return 2
    records = read_records(args.input)
    features = [features_from_record(record) for record in records]
    x = np.array([[item[name] for name in FEATURE_NAMES] for item in features])
    y = np.array([_wrong(record) for record in records], dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        print("fit_gate: the input needs both correct and wrong answers")
        return 2
    logits, folds = cross_validate(x, y, args.folds, args.seed, args.l2)
    score = auroc(logits, y)
    a, b = _fit_platt(logits, y)
    calibrated = _sigmoid(a * logits + b)
    write_reliability(args.figure, calibrated, y)
    weights, intercept = _fit_logistic(x, y, args.l2)
    model = {
        "format_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "coefficients": [float(value) for value in weights],
        "intercept": intercept,
        "calibration": {"kind": "platt", "a": a, "b": b},
        "training_split": args.split,
        "manifest_hash": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "metrics": {"auroc_oof": score, "auroc_calibrated": auroc(calibrated, y),
                    "n": len(y), "wrong": int(y.sum()), "folds": folds},
    }
    Path(args.out).write_text(json.dumps(model, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(f"fit_gate: n={len(y)} wrong={int(y.sum())} folds={folds} "
          f"auroc_oof={score:.3f} auroc_calibrated={model['metrics']['auroc_calibrated']:.3f}")
    if score < args.min_auroc:
        print(f"fit_gate: AUROC {score:.3f} is below --min-auroc {args.min_auroc:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
