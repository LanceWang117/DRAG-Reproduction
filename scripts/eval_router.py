"""Evaluate a fitted J20 router on logged multi-arm runs: Pareto curve vs random routing.

Input is the same JSONL the fitter reads: one record per ``(question_id, route)`` with the
four pre-debate features, ``correct``/``wrong``, and an optional measured ``cost``. The
script sweeps the cost weight ``lambda``, chooses a route per question, and plots accuracy
against mean cost. The random baseline keeps the router's own route frequencies (the
"equal DRAG share" control): every question samples a route from those frequencies, over
1,000 seeds by default, and the plot shows the mean with a 2.5/97.5 percentile bar.

The figure is a dependency-free SVG because ``requirements-ci.txt`` has no Matplotlib and
J20 does not authorize a new dependency; this matches ``fit_gate.py``'s reliability SVG.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from drag_plus.gates.router import ROUTE_NAMES, decide_route, load_router
from drag_plus.gates.router_features import ROUTER_FEATURE_NAMES

DEFAULT_LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1)


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


def _features(record):
    source = record.get("features")
    if not isinstance(source, dict) or set(source) != set(ROUTER_FEATURE_NAMES):
        raise ValueError(
            f"features must be an object with exactly {list(ROUTER_FEATURE_NAMES)}")
    values = {name: float(source[name]) for name in ROUTER_FEATURE_NAMES}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("every feature must be finite")
    return values


def _correct(record):
    if "correct" in record:
        return bool(record["correct"])
    if "wrong" in record:
        return not bool(record["wrong"])
    raise ValueError("a record needs correct or wrong")


def group_questions(records, model):
    """One entry per question with every route's features, correctness and total cost."""
    grouped = {}
    for record in records:
        route = str(record.get("route", ""))
        if route not in ROUTE_NAMES:
            raise ValueError(f"unknown route {route!r}")
        qid = str(record.get("question_id", record.get("id", "")))
        if not qid:
            raise ValueError("a record needs question_id or id")
        measured = record.get("cost")
        base = (float(measured) if isinstance(measured, (int, float))
                and not isinstance(measured, bool) else model.costs[route])
        grouped.setdefault(qid, {})[route] = {
            "features": _features(record), "correct": _correct(record),
            "cost": model.feature_cost + base}
    for qid, routes in grouped.items():
        if set(routes) != set(ROUTE_NAMES):
            raise ValueError(f"question {qid!r} does not have all routes")
    return [{"qid": qid, **routes} for qid, routes in sorted(grouped.items())]


def _shares(routes):
    counts = {route: 0 for route in ROUTE_NAMES}
    for choice in routes:
        counts[choice] += 1
    return {route: counts[route] / len(routes) for route in ROUTE_NAMES}


def router_points(questions, model, lambdas):
    """One accuracy/cost point per cost weight, with the chosen route frequencies."""
    points = []
    for lam in lambdas:
        choices = [decide_route(q[ROUTE_NAMES[0]]["features"], model, lam).route
                   for q in questions]
        points.append({
            "lambda": float(lam),
            "accuracy": sum(q[route]["correct"] for q, route in zip(questions, choices))
                        / len(questions),
            "cost": sum(q[route]["cost"] for q, route in zip(questions, choices))
                    / len(questions),
            "shares": _shares(choices)})
    return points


def _sample(questions, shares, rng):
    choices = rng.choice(np.array(ROUTE_NAMES), size=len(questions),
                         p=np.array([shares[route] for route in ROUTE_NAMES]))
    accuracy = sum(q[route]["correct"] for q, route in zip(questions, choices)) / len(questions)
    cost = sum(q[route]["cost"] for q, route in zip(questions, choices)) / len(questions)
    return float(accuracy), float(cost)


def random_points(questions, points, seeds, seed):
    """Mean and 2.5/97.5 percentile interval of the equal-share random baseline."""
    rng = np.random.default_rng(seed)
    baseline = []
    for point in points:
        draws = np.array([_sample(questions, point["shares"], rng) for _ in range(seeds)])
        baseline.append({
            "accuracy_mean": float(draws[:, 0].mean()),
            "accuracy_low": float(np.percentile(draws[:, 0], 2.5)),
            "accuracy_high": float(np.percentile(draws[:, 0], 97.5)),
            "cost_mean": float(draws[:, 1].mean()), "seeds": int(seeds)})
    return baseline


def evaluate(records, model, lambdas=DEFAULT_LAMBDAS, seeds=1000, seed=0):
    """Router and random points plus a dominance flag for each cost weight."""
    questions = group_questions(records, model)
    router = router_points(questions, model, lambdas)
    baseline = random_points(questions, router, seeds, seed)
    for chosen, random in zip(router, baseline):
        random["dominates"] = (chosen["accuracy"] >= random["accuracy_mean"]
                               and chosen["cost"] <= random["cost_mean"])
    return {"router": router, "random": baseline, "questions": len(questions)}


def _xy(cost, accuracy, costs, width, height, pad):
    low, high = min(costs), max(costs)
    span = (high - low) or 1.0
    x = pad + (width - 2 * pad) * (cost - low) / span
    y = height - pad - (height - 2 * pad) * accuracy
    return f"{x:.1f},{y:.1f}"


def write_svg(path, result):
    """Dependency-free accuracy-vs-cost Pareto plot with the random-baseline bars."""
    width, height, pad = 520, 360, 48
    router, baseline = result["router"], result["random"]
    costs = [point["cost"] for point in router] + [point["cost_mean"] for point in baseline]
    curve = " ".join(_xy(point["cost"], point["accuracy"], costs, width, height, pad)
                     for point in router)
    circles = ""
    for point in router:
        x, y = _xy(point["cost"], point["accuracy"], costs, width, height, pad).split(",")
        circles += f'<circle cx="{x}" cy="{y}" r="4" fill="#1f77b4"/>'
    bars = ""
    for point in baseline:
        low = _xy(point["cost_mean"], point["accuracy_low"], costs, width, height, pad)
        high = _xy(point["cost_mean"], point["accuracy_high"], costs, width, height, pad)
        x, y = _xy(point["cost_mean"], point["accuracy_mean"], costs, width, height, pad).split(",")
        bars += (f'<line x1="{x}" y1="{low.split(",")[1]}" x2="{x}" y2="{high.split(",")[1]}" '
                 f'stroke="#d62728" stroke-width="1"/>'
                 f'<circle cx="{x}" cy="{y}" r="3" fill="#d62728"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'<text x="{pad}" y="22" font-family="sans-serif" font-size="13">'
           f'Router Pareto vs random routing (equal share, {result["questions"]} questions)'
           f'</text>'
           f'<polyline points="{curve}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
           f'{circles}{bars}'
           f'<text x="{width // 2}" y="{height - 8}" font-family="sans-serif" font-size="11" '
           f'text-anchor="middle">mean cost per question (relative)</text>'
           f'<text x="12" y="{height // 2}" font-family="sans-serif" font-size="11" '
           f'transform="rotate(-90 12 {height // 2})" text-anchor="middle">accuracy</text>'
           f'</svg>\n')
    Path(path).write_text(svg, encoding="utf-8")


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.seeds < 1:
        print("eval_router: --seeds must be at least 1")
        return 2
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    model = load_router(args.model)
    result = evaluate(read_records(args.input), model, lambdas, args.seeds, args.seed)
    write_svg(args.figure, result)
    for chosen, random in zip(result["router"], result["random"]):
        print(f"eval_router: lambda={chosen['lambda']:.3f} "
              f"router=({chosen['cost']:.2f}, {chosen['accuracy']:.3f}) "
              f"random=({random['cost_mean']:.2f}, {random['accuracy_mean']:.3f} "
              f"[{random['accuracy_low']:.3f}, {random['accuracy_high']:.3f}]) "
              f"dominates={random['dominates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
