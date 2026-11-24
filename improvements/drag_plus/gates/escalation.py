"""Fitted escalation risk and lazy-Opponent sufficiency (J14).

The model is a small JSON document: feature names, logistic coefficients, an
intercept, a calibration map, the training split and the input manifest hash. The
loader fails closed: a model trained on the ``test`` split, an unknown feature name
or an unknown calibration kind is rejected rather than silently approximated.

``risk`` is ``P(wrong | label-free features)``. The answer gate stops when
``risk <= tau``; the lazy Opponent skips the query debate when the sufficiency
probability is above ``tau``. Both thresholds are probabilities in ``[0, 1]``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .. import prompts
from ..judge.logprob import ScoredOutput
from .features import FEATURE_NAMES

SUFFICIENCY_LABELS: Tuple[str, str] = ("sufficient", "insufficient")
MAX_GATE_ROUNDS = 3
_SUFFICIENCY_SYSTEM = (
    "You are a retrieval sufficiency judge. Decide whether the retrieved evidence is "
    "sufficient to answer the question reliably. Reply with exactly one word: "
    "sufficient or insufficient.")


class GateModelError(ValueError):
    """A gate model is missing, malformed, test-split or incompatible with the code."""


@dataclass(frozen=True)
class GateModel:
    """A validated logistic gate plus its probability calibration."""

    coefficients: Tuple[float, ...]
    intercept: float
    calibration: Mapping[str, Any]
    training_split: str
    manifest_hash: str


@dataclass(frozen=True)
class GateDecision:
    """One answer-stage stopping decision: the calibrated risk and ``stop`` bit."""

    risk: float
    stop: bool
    features: Mapping[str, float]


@dataclass(frozen=True)
class SufficiencyDecision:
    """The lazy-Opponent decision; ``None`` score means the baseline path runs."""

    score: Optional[float]
    skip: bool
    status: str


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateModelError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise GateModelError(f"{name} must be finite, got {value!r}")
    return number


def _calibration(data: Any) -> Mapping[str, Any]:
    if data is None:
        return {"kind": "identity"}
    if not isinstance(data, dict):
        raise GateModelError("calibration must be an object")
    kind = data.get("kind", "identity")
    if kind == "identity":
        return {"kind": "identity"}
    if kind == "platt":
        return {"kind": "platt", "a": _finite(data.get("a"), "calibration.a"),
                "b": _finite(data.get("b"), "calibration.b")}
    if kind == "isotonic":
        xs, ys = data.get("x"), data.get("y")
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys) \
                or len(xs) < 2:
            raise GateModelError("isotonic calibration needs equal-length x and y lists")
        x = [_finite(value, "calibration.x") for value in xs]
        y = [_finite(value, "calibration.y") for value in ys]
        if any(right <= left for left, right in zip(x, x[1:])):
            raise GateModelError("isotonic calibration.x must be strictly increasing")
        return {"kind": "isotonic", "x": x, "y": y}
    raise GateModelError(f"unknown calibration kind {kind!r}")


def load_model(path: Any) -> GateModel:
    """Read and validate one fitted gate JSON; any mismatch raises ``GateModelError``."""
    if path is None or not str(path).strip():
        raise GateModelError(
            "escalation_gate='risk' needs escalation_model pointing at a fitted JSON")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateModelError(f"cannot read gate model {str(path)!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise GateModelError("gate model must be a JSON object")
    split = str(data.get("training_split", ""))
    if split == "test":
        raise GateModelError("refusing a gate model trained on the test split")
    names = tuple(data.get("feature_names") or ())
    if names != FEATURE_NAMES:
        raise GateModelError(
            f"gate model features {names!r} do not match code {FEATURE_NAMES!r}")
    coefficients = data.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != len(FEATURE_NAMES):
        raise GateModelError("coefficients must be the same length as feature_names")
    return GateModel(
        coefficients=tuple(_finite(value, "coefficient") for value in coefficients),
        intercept=_finite(data.get("intercept"), "intercept"),
        calibration=_calibration(data.get("calibration")),
        training_split=split,
        manifest_hash=str(data.get("manifest_hash", "")))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _vector(features: Mapping[str, float]) -> Tuple[float, ...]:
    values = []
    for name in FEATURE_NAMES:
        if name not in features:
            raise GateModelError(f"missing gate feature {name!r}")
        values.append(_finite(features[name], f"feature {name}"))
    return tuple(values)


def _isotonic(value: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation with clamped ends (``numpy.interp`` semantics)."""
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            span = xs[index] - xs[index - 1]
            weight = (value - xs[index - 1]) / span
            return ys[index - 1] + weight * (ys[index] - ys[index - 1])
    return ys[-1]


def calibrated_risk(features: Mapping[str, float], model: GateModel) -> float:
    """``P(wrong)`` after the model's calibration map, in ``[0, 1]``."""
    logit = model.intercept + sum(weight * value for weight, value
                                  in zip(model.coefficients, _vector(features)))
    kind = model.calibration["kind"]
    if kind == "platt":
        logit = model.calibration["a"] * logit + model.calibration["b"]
        return _sigmoid(logit)
    if kind == "isotonic":
        return _isotonic(logit, model.calibration["x"], model.calibration["y"])
    return _sigmoid(logit)


def decide(features: Mapping[str, float], model: GateModel, tau: float) -> GateDecision:
    """Stop when the calibrated wrong-answer risk is at most ``tau`` (ties stop)."""
    threshold = _finite(tau, "escalation_tau")
    if not 0.0 <= threshold <= 1.0:
        raise GateModelError(f"escalation_tau must be in [0, 1], got {threshold!r}")
    risk = calibrated_risk(features, model)
    return GateDecision(risk=risk, stop=risk <= threshold, features=dict(features))


def sufficiency_messages(question: str, query_pool) -> list:
    """The one-shot sufficiency prompt: question plus the current evidence pool."""
    return [{"role": "system", "content": _SUFFICIENCY_SYSTEM},
            {"role": "user", "content": prompts.query_round_user(question, query_pool)["content"]}]


def sufficiency_probability(scored: ScoredOutput) -> Optional[float]:
    """Softmax of the ``sufficient`` label against ``insufficient``; None if missing."""
    scores = scored.label_logprobs
    good, bad = scores.get("sufficient"), scores.get("insufficient")
    if good is None or bad is None or not math.isfinite(good) or not math.isfinite(bad):
        return None
    high = max(good, bad)
    good_exp, bad_exp = math.exp(good - high), math.exp(bad - high)
    return good_exp / (good_exp + bad_exp)


def resolve_sufficiency(scored: ScoredOutput, tau: float) -> SufficiencyDecision:
    """Skip the Opponent only when the sufficiency probability is strictly above tau."""
    probability = sufficiency_probability(scored)
    if probability is None:
        return SufficiencyDecision(None, False, "unavailable")
    skip = probability > float(tau)
    return SufficiencyDecision(probability, skip, "sufficient" if skip else "insufficient")


def run_lazy_opponent(ctx, state, round_index: int) -> bool:
    """Score retention sufficiency once; True ends the query stage before the agents.

    One scored call is made with the J12 label scorer. A generator without that
    capability logs ``unavailable`` and returns False, so the baseline Opponent runs.
    """
    if not ctx.flags.lazy_query_opponent:
        return False
    prompt = ctx.backends.render(sufficiency_messages(state.question, state.query_pool))
    scorer = getattr(ctx.backends.generator, "generate_with_scores", None)
    if callable(scorer):
        scored = ctx.backends.complete_with_scores(
            [prompt], labels=SUFFICIENCY_LABELS,
            call=state.call("query_sufficiency", "judge"))
        decision = resolve_sufficiency(scored[0], ctx.flags.escalation_tau)
    else:
        decision = SufficiencyDecision(None, False, "unavailable")
    state.sufficiency_score = decision.score
    key = f"QueryStage_Sufficiency_Round{round_index}"
    if decision.score is not None:
        state.emit(f"{key}_Score", decision.score)
    state.emit(f"{key}_Decision", decision.status)
    state.decisions.append({"stage": "query", "round": round_index,
                            "decision": "lazy_opponent", "status": decision.status,
                            "score": decision.score})
    return decision.skip
