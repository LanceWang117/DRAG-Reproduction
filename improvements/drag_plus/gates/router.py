"""Fit-free loading, cost-aware decisions and pre-debate routing for J20.

The model JSON holds one logistic correctness model per route, each route's relative
cost, the feature-extraction cost, the cost weight ``lambda`` and the training split.
The decision is ``argmax_r [P(correct | r, q) - lambda * cost(r)]``; ``cost(r)``
includes the retrieval, Naive generation and Judge probe paid before the route runs.
The loader fails closed on a test split, feature drift, an unknown route or a bad cost.

``run_routed`` retrieves once, computes the four features, records the decision and
executes the chosen path. A null ``router`` flag returns ``False`` so the baseline
pipeline runs unchanged.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .. import prompts
from ..analysis.extract import extract_answer
from ..answer_stage import run_answer_only, run_answer_stage
from ..answers import native_pred, unified_pred
from ..evidence.actions import prepare_pool, search, sync_pool
from ..evidence.store import register_query
from ..judge.action_schema import active_queries
from ..judge.logprob import LABELS, resolve_scored
from ..query_stage import run_query_stage
from .router_features import (ROUTER_FEATURE_NAMES, RouterInputs, compute_features,
                              extractive_answer, top_passages)

ROUTE_NAMES: Tuple[str, ...] = ("naive", "refine_only", "full_drag")
FORMAT_VERSION = 1


class RouterModelError(ValueError):
    """A router model is missing, malformed, test-split or incompatible with the code."""


@dataclass(frozen=True)
class RouterModel:
    """One validated per-route correctness model plus its cost metadata."""

    weights: Mapping[str, Tuple[float, ...]]
    intercepts: Mapping[str, float]
    costs: Mapping[str, float]
    feature_cost: float
    cost_weight: float
    training_split: str
    manifest_hash: str


@dataclass(frozen=True)
class RouterDecision:
    """One route choice with its probabilities, utilities, features and costs."""

    route: str
    probabilities: Mapping[str, float]
    utilities: Mapping[str, float]
    features: Mapping[str, float]
    costs: Mapping[str, float]
    cost_weight: float


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouterModelError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise RouterModelError(f"{name} must be finite, got {value!r}")
    return number


def _weights(route: str, data: Any) -> Tuple[Tuple[float, ...], float]:
    if not isinstance(data, dict):
        raise RouterModelError(f"route {route!r} must be an object")
    coefficients = data.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != len(ROUTER_FEATURE_NAMES):
        raise RouterModelError(
            f"route {route!r} needs {len(ROUTER_FEATURE_NAMES)} coefficients")
    return (tuple(_finite(value, f"{route}.coefficient") for value in coefficients),
            _finite(data.get("intercept"), f"{route}.intercept"))


def _costs(data: Any) -> Dict[str, float]:
    if not isinstance(data, dict):
        raise RouterModelError("costs must be an object")
    costs = {route: _finite(data.get(route), f"costs.{route}") for route in ROUTE_NAMES}
    if any(value < 0.0 for value in costs.values()):
        raise RouterModelError("costs must be non-negative")
    return costs


def load_router(path: Any) -> RouterModel:
    """Read and validate one fitted router JSON; any mismatch raises ``RouterModelError``."""
    if path is None or not str(path).strip():
        raise RouterModelError("router needs a fitted model path")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RouterModelError(f"cannot read router model {str(path)!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise RouterModelError("router model must be a JSON object")
    if data.get("format_version") != FORMAT_VERSION:
        raise RouterModelError("unsupported router format_version")
    split = str(data.get("training_split", ""))
    if split == "test":
        raise RouterModelError("refusing a router trained on the test split")
    names = tuple(data.get("feature_names") or ())
    if names != ROUTER_FEATURE_NAMES:
        raise RouterModelError(
            f"router features {names!r} do not match code {ROUTER_FEATURE_NAMES!r}")
    routes = data.get("routes")
    if not isinstance(routes, dict) or set(routes) != set(ROUTE_NAMES):
        raise RouterModelError(f"routes must define exactly {list(ROUTE_NAMES)}")
    pairs = {route: _weights(route, routes[route]) for route in ROUTE_NAMES}
    feature_cost = _finite(data.get("feature_cost", 0.0), "feature_cost")
    cost_weight = _finite(data.get("cost_weight", 0.0), "cost_weight")
    if feature_cost < 0.0 or cost_weight < 0.0:
        raise RouterModelError("feature_cost and cost_weight must be non-negative")
    return RouterModel(weights={route: pair[0] for route, pair in pairs.items()},
                       intercepts={route: pair[1] for route, pair in pairs.items()},
                       costs=_costs(data.get("costs")), feature_cost=feature_cost,
                       cost_weight=cost_weight, training_split=split,
                       manifest_hash=str(data.get("manifest_hash", "")))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _vector(features: Mapping[str, float]) -> Tuple[float, ...]:
    values = []
    for name in ROUTER_FEATURE_NAMES:
        if name not in features:
            raise RouterModelError(f"missing router feature {name!r}")
        values.append(_finite(features[name], f"feature {name}"))
    return tuple(values)


def decide_route(features: Mapping[str, float], model: RouterModel,
                 cost_weight: Optional[float] = None) -> RouterDecision:
    """``argmax_r P(correct|r) - lambda*cost(r)``; ties keep the cheaper listed route."""
    lam = model.cost_weight if cost_weight is None else _finite(cost_weight, "cost_weight")
    if lam < 0.0:
        raise RouterModelError(f"cost_weight must be non-negative, got {lam!r}")
    vector = _vector(features)
    probabilities = {route: _sigmoid(model.intercepts[route] + sum(
        weight * value for weight, value in zip(model.weights[route], vector)))
        for route in ROUTE_NAMES}
    costs = {route: model.feature_cost + model.costs[route] for route in ROUTE_NAMES}
    utilities = {route: probabilities[route] - lam * costs[route] for route in ROUTE_NAMES}
    route = min(ROUTE_NAMES, key=lambda name: (-utilities[name], ROUTE_NAMES.index(name)))
    return RouterDecision(route, probabilities, utilities, dict(features), costs, lam)


def _retrieve_original(ctx, state) -> None:
    """The pre-debate retrieval; the query stage reuses this non-empty pool."""
    state.round = 0
    prepare_pool(ctx, state)
    question = state.question.strip()
    passages = search(ctx, state, state.question)
    if state.evidence is not None:
        state.evidence.add_query(question, passages, 0, anchored=True)
        sync_pool(ctx, state)
    else:
        state.query_pool[question] = passages
    for query in active_queries(state):
        register_query(state.queries, query, "original", 0)

def _naive_raw(ctx, state) -> str:
    """One Naive generation on the original retrieval; reused by the naive route."""
    passages = top_passages(state.query_pool)
    prompt = ctx.backends.prompt_template.get_string(
        question=state.question, retrieval_result=passages)
    return ctx.backends.complete(prompt, call=state.call("router_naive", "naive"))


def _judge_probe(ctx, state, own_answer: str, retrieval_answer: str) -> Optional[float]:
    """One J12 scored Judge call on the two pre-debate answers; None when unavailable."""
    if not own_answer or not retrieval_answer:
        return None
    if not callable(getattr(ctx.backends.generator, "generate_with_scores", None)):
        return None
    outputs = {"Proponent Agent 0": retrieval_answer, "Opponent Agent 0": own_answer}
    prompt = ctx.backends.render(
        [prompts.query_moderator_system(outputs, state.question, state.query_pool)])
    scored = ctx.backends.complete_with_scores(
        [prompt], labels=LABELS, call=state.call("router_judge", "judge"))
    return resolve_scored(scored, 0.0, False).score


def _record(state, decision: RouterDecision, probe: Optional[float]) -> None:
    state.router_route = decision.route
    state.router_features = dict(decision.features)
    state.router_cost = decision.costs[decision.route]
    state.emit("Router_Features", dict(decision.features))
    state.emit("Router_Route", decision.route)
    state.emit("Router_RouteCost", decision.costs[decision.route])
    state.decisions.append({
        "stage": "router", "round": 0, "decision": "route", "status": decision.route,
        "features": dict(decision.features), "probabilities": dict(decision.probabilities),
        "cost": decision.costs[decision.route], "judge_probe": probe})


def _finish_naive(ctx, state, raw: str) -> None:
    state.raw_pred = raw
    if ctx.flags.answer_extractor == "native":
        state.pred = native_pred("naive", raw)
        state.emit("raw_pred", raw)
        state.emit("pred", state.pred)
    else:
        state.pred = unified_pred(raw, ctx.dataset_name)
    state.stage = "answer"


def _execute_route(ctx, state, route: str, raw: str) -> None:
    if route == "naive":
        _finish_naive(ctx, state, raw)
        return
    run_query_stage(ctx, state)  # round 0 sees the non-empty pool and skips retrieval
    state.emit("QueryStage_QueryPool", state.query_pool)
    if route == "refine_only":
        run_answer_only(ctx, state)
    else:
        run_answer_stage(ctx, state)


def run_routed(ctx, state) -> bool:
    """Run the router path; ``False`` means the flag is off and the baseline must run."""
    if not ctx.flags.router:
        return False
    model = load_router(ctx.flags.router)
    _retrieve_original(ctx, state)
    raw = _naive_raw(ctx, state)
    own_answer = extract_answer(raw, ctx.strategy_qa)
    retrieval_answer = extractive_answer(top_passages(state.query_pool))
    probe = _judge_probe(ctx, state, own_answer, retrieval_answer)
    features = compute_features(RouterInputs(
        passages=top_passages(state.query_pool), judge_score=probe,
        own_answer=own_answer, retrieval_answer=retrieval_answer))
    decision = decide_route(features, model, ctx.flags.router_cost_weight)
    _record(state, decision, probe)
    _execute_route(ctx, state, decision.route, raw)
    return True
