#!/usr/bin/env python3
"""Offline throughput model for the refill window and lockstep batches (J16).

The fake backend charges ``a + b * batch_size`` seconds per dispatched call: ``a`` is
the fixed call overhead and ``b`` the marginal cost of one more question in the batch.
A step dispatches every active question, so its batch size is the active count. Each
question needs a number of steps drawn from a deterministic skewed distribution; a
question leaves as soon as it reaches zero. Refill admits a pending question the moment
a slot frees, while lockstep runs fixed chunks of ``window`` questions to completion.

The output is structural only: steps, makespan, mean occupancy as a fraction of the
window, and modelled questions/hour. It is not a measurement and writes no run logs.

Usage:
    python scripts/simulate_throughput.py --questions 64 --window 8
    python scripts/simulate_throughput.py --questions 64 --window 8 --json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class CostModel:
    """Per-call model: ``fixed + per_item * batch_size`` seconds."""

    fixed: float = 0.5
    per_item: float = 0.02

    def call(self, batch: int) -> float:
        """Seconds one dispatched call with ``batch`` questions costs."""
        return self.fixed + self.per_item * batch


@dataclass(frozen=True)
class SimulationResult:
    """Structural outcome; ``mean_occupancy`` is a fraction of ``window``."""

    policy: str
    window: int
    questions: int
    steps: int
    makespan_seconds: float
    mean_occupancy: float
    questions_per_hour: float
    max_occupancy: float


def skewed_profile(count: int, seed: int = 7, shape: float = 1.5,
                   cap: int = 16) -> List[int]:
    """Deterministic Pareto step counts: mostly short, with a long tail.

    ``shape`` is the Pareto alpha: smaller values make the tail heavier. The result is
    clamped to ``[1, cap]`` so the simulation is bounded and reproducible.
    """
    rng = random.Random(seed)
    return [max(1, min(cap, int(round(rng.paretovariate(shape))))) for _ in range(count)]


def _finish(result: SimulationResult, profile: Sequence[int]) -> SimulationResult:
    """Attach the modelled questions/hour once the makespan is known."""
    rate = len(profile) / result.makespan_seconds * 3600 if result.makespan_seconds else 0.0
    return SimulationResult(**{**asdict(result), "questions_per_hour": rate})


def simulate_lockstep(profile: Sequence[int], window: int,
                      cost: CostModel) -> SimulationResult:
    """Fixed chunks of ``window``; no question is admitted until the chunk finishes."""
    steps = 0
    makespan = 0.0
    active_total = 0
    counted_steps = 0
    peak = 0
    for start in range(0, len(profile), window):
        active = list(profile[start:start + window])
        more_pending = start + window < len(profile)
        while active:
            peak = max(peak, len(active))
            makespan += cost.call(len(active))
            steps += 1
            if more_pending:
                active_total += len(active)
                counted_steps += 1
            active = [left - 1 for left in active if left > 1]
    occupancy = active_total / (counted_steps * window) if counted_steps else 0.0
    result = SimulationResult("lockstep", window, len(profile), steps, makespan,
                              occupancy, 0.0, peak / window if window else 0.0)
    return _finish(result, profile)


def simulate_refill(profile: Sequence[int], window: int,
                    cost: CostModel) -> SimulationResult:
    """Windowed admission: a finished question's slot is refilled immediately."""
    pending = list(profile)
    active = [pending.pop(0) for _ in range(min(window, len(pending)))]
    steps = 0
    makespan = 0.0
    active_total = 0
    counted_steps = 0
    peak = 0
    while active:
        peak = max(peak, len(active))
        makespan += cost.call(len(active))
        steps += 1
        if pending:
            active_total += len(active)
            counted_steps += 1
        active = [left - 1 for left in active if left > 1]
        while len(active) < window and pending:
            active.append(pending.pop(0))
    occupancy = active_total / (counted_steps * window) if counted_steps else 0.0
    result = SimulationResult("refill", window, len(profile), steps, makespan,
                              occupancy, 0.0, peak / window if window else 0.0)
    return _finish(result, profile)


def simulate(profile: Sequence[int], window: int,
             cost: CostModel) -> List[SimulationResult]:
    """Both policies on the same profile, lockstep first."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if not profile:
        raise ValueError("profile must not be empty")
    return [simulate_lockstep(profile, window, cost),
            simulate_refill(profile, window, cost)]


def _format(results: Sequence[SimulationResult]) -> str:
    lines = [f"questions={results[0].questions} window={results[0].window}",
             "policy    steps  makespan_s  mean_occupancy  questions_per_hour"]
    for result in results:
        lines.append(f"{result.policy:<9} {result.steps:>5}  "
                     f"{result.makespan_seconds:>10.3f}  "
                     f"{result.mean_occupancy:>14.3f}  "
                     f"{result.questions_per_hour:>19.2f}")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shape", type=float, default=1.5,
                        help="Pareto alpha; smaller is more skewed")
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--fixed", type=float, default=0.5, dest="fixed_cost")
    parser.add_argument("--per-item", type=float, default=0.02, dest="per_item")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    profile = skewed_profile(args.questions, args.seed, args.shape, args.cap)
    cost = CostModel(args.fixed_cost, args.per_item)
    results = simulate(profile, args.window, cost)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print(_format(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
