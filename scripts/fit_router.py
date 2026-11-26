"""Fit per-route correctness models for the J20 cost-aware router.

Input is JSONL with one record per ``(question_id, route)``: the four pre-debate feature
values (or nothing, in which case the record is rejected), ``correct``/``wrong``, and an
optional measured ``cost``. Gold labels are used only as the target. The script refuses
the ``test`` split, fits one L2 logistic model per route, reports leave-one-dataset-out
(LODO) AUROC, and writes the small model JSON that ``gates/router.py`` loads.

Costs are the mean measured cost per route when the input records carry one, else the
documented defaults. ``feature_cost`` is the shared retrieval + Naive generation + Judge
probe paid before every route; it is recorded separately so the decision can add it once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from drag_plus.gates.router import FORMAT_VERSION, ROUTE_NAMES
from drag_plus.gates.router_features import ROUTER_FEATURE_NAMES

DEFAULT_COSTS = {"naive": 1.0, "refine_only": 4.0, "full_drag": 10.0}


def read_records(path):
    """Parse a JSONL file into objects, rejecting blank-only and non-object lines."""
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


def _features(record):
    """The four whitelisted features, in code order, all finite."""
    source = record.get("features")
    if not isinstance(source, dict) or set(source) != set(ROUTER_FEATURE_NAMES):
        raise ValueError(
            f"features must be an object with exactly {list(ROUTER_FEATURE_NAMES)}")
    values = [float(source[name]) for name in ROUTER_FEATURE_NAMES]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("every feature must be finite")
    return values


def _route(record):
    route = str(record.get("route", ""))
    if route not in ROUTE_NAMES:
        raise ValueError(f"unknown route {route!r}")
    return route


def _correct(record):
    if "correct" in record:
        return bool(record["correct"])
    if "wrong" in record:
        return not bool(record["wrong"])
    raise ValueError("a record needs correct or wrong; gold text is not parsed here")


def _dataset(record):
    return str(record.get("dataset") or record.get("source") or "all")


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _fit_logistic(x, y, l2=1.0, steps=800, lr=0.5):
    """Small L2 logistic regression by full-batch gradient descent; no sklearn."""
    weights, bias = np.zeros(x.shape[1]), 0.0
    for _ in range(steps):
        error = _sigmoid(x @ weights + bias) - y
        weights -= lr * (x.T @ error / len(y) + l2 * weights / len(y))
        bias -= lr * float(np.mean(error))
    return weights, float(bias)


def _prior(y):
    probability = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    return np.zeros(len(ROUTER_FEATURE_NAMES)), math.log(probability / (1 - probability))


def _fit_route(rows, l2):
    """Fit one route; a single-class training slice becomes an intercept-only prior."""
    x = np.array([row[0] for row in rows], dtype=float)
    y = np.array([row[1] for row in rows], dtype=float)
    return _fit_logistic(x, y, l2) if 0 < y.sum() < len(y) else _prior(y)


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


def auroc(scores, labels):
    """Rank AUROC for ``correct == 1``; ties share their average rank."""
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = _ties_average_ranks(np.asarray(scores, dtype=float))
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def _records_by_route(records):
    grouped = {route: [] for route in ROUTE_NAMES}
    for record in records:
        route = _route(record)
        grouped[route].append((_features(record), 1.0 if _correct(record) else 0.0, record))
    return grouped


def _lodo(records, l2):
    """Leave one dataset out: fit every route on the other datasets, score the held one."""
    datasets = sorted({_dataset(record) for record in records})
    report = {}
    for held in datasets:
        train = [record for record in records if _dataset(record) != held]
        test = [record for record in records if _dataset(record) == held]
        grouped_train, grouped_test = _records_by_route(train), _records_by_route(test)
        scores = {}
        for route in ROUTE_NAMES:
            weights, bias = _fit_route(grouped_train[route], l2)
            rows = grouped_test[route]
            logits = [float(np.dot(weights, row[0]) + bias) for row in rows]
            scores[route] = auroc(logits, [row[1] for row in rows])
        report[held] = scores
    return report


def _mean_costs(grouped):
    costs = {}
    for route in ROUTE_NAMES:
        values = [float(record.get("cost")) for _, _, record in grouped[route]
                  if isinstance(record.get("cost"), (int, float))
                  and not isinstance(record.get("cost"), bool)]
        costs[route] = sum(values) / len(values) if values else DEFAULT_COSTS[route]
    return costs


def _model(records, args):
    grouped = _records_by_route(records)
    routes = {}
    for route in ROUTE_NAMES:
        weights, bias = _fit_route(grouped[route], args.l2)
        routes[route] = {"coefficients": [float(value) for value in weights],
                         "intercept": bias}
    lodo = _lodo(records, args.l2)
    macro = float(np.mean([score for scores in lodo.values() for score in scores.values()]))
    costs = _mean_costs(grouped)
    return {
        "format_version": FORMAT_VERSION,
        "feature_names": list(ROUTER_FEATURE_NAMES),
        "routes": routes,
        "costs": costs,
        "feature_cost": args.feature_cost,
        "cost_weight": args.cost_weight,
        "training_split": args.split,
        "manifest_hash": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "metrics": {"lodo_auroc": lodo, "lodo_macro_auroc": macro,
                    "n": len(records), "costs": costs},
    }


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--feature-cost", type=float, default=3.0)
    parser.add_argument("--cost-weight", type=float, default=0.02)
    parser.add_argument("--min-lodo-auroc", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.split == "test":
        print("fit_router: refusing to fit on the test split")
        return 2
    if args.feature_cost < 0 or args.cost_weight < 0:
        print("fit_router: costs and the cost weight must be non-negative")
        return 2
    model = _model(read_records(args.input), args)
    Path(args.out).write_text(json.dumps(model, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    macro = model["metrics"]["lodo_macro_auroc"]
    print(f"fit_router: n={model['metrics']['n']} lodo_macro_auroc={macro:.3f} "
          f"costs={model['metrics']['costs']}")
    if macro < args.min_lodo_auroc:
        print(f"fit_router: LODO AUROC {macro:.3f} is below --min-lodo-auroc "
              f"{args.min_lodo_auroc:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
