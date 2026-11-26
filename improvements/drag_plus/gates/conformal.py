"""Split-conformal selective answering (J21).

Nonconformity is ``s(x, y) = 1 - p_hat(y | x)`` over the debate's answer clusters.
``p_hat`` is a softmax of a smoothed vote share plus evidence support, sharpened by the
query-stage Judge margin; that margin compares queries, so it is a temperature, never
a per-label vote.

Calibration is Mondrian (per dataset): ``ceil((n + 1)(1 - alpha)) / n`` quantile
(Angelopoulos & Bates, 2021); a rank above ``n`` gives ``q_hat = 1`` and abstains.
Below ``min_n`` a dataset uses the pooled threshold and is flagged. Files record their
split, ``test`` is refused, and ``conformal_alpha = null`` is a no-op (T0).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..analysis.extract import extract_answer
from .features import (agent_answers, cluster_key as normalise_answer,
                       evidence_support, passage_texts, s_bar_o)

FORMAT_VERSION = 1
VOTE_SMOOTH = 0.5
EVIDENCE_WEIGHT = 1.0
JUDGE_WEIGHT = 1.0


class CalibrationError(ValueError):
    """A conformal calibration file is missing, malformed, mismatched or test-split."""


@dataclass(frozen=True)
class Calibration:
    alpha: float
    min_n: int
    pooled_n: int
    pooled_qhat: float
    datasets: Mapping[str, Tuple[int, float]]
    split: str


def cluster_key(text: str, yes_no: bool = False) -> str:
    """E(.) extraction plus the J01 token normalisation: one key per answer cluster."""
    extracted = extract_answer(text or "", yes_no=yes_no)
    return normalise_answer(extracted) or extracted.strip().lower()


def nonconformity(probability: float) -> float:
    """``1 - p_hat``, clamped so a bad caller cannot produce an out-of-range score."""
    return 1.0 - min(1.0, max(0.0, float(probability)))


def probability_distribution(votes: Mapping[str, int], supports: Mapping[str, float],
                             judge_score: Optional[float]) -> Dict[str, float]:
    """Softmax over ``cluster_key`` slots from votes, supports and Judge decisiveness.

    A cluster with support but no vote is kept: evidence can raise an answer the agents
    did not repeat. No votes means nothing to score, so the result is empty; no Judge
    score is neutral and leaves the temperature at 1.
    """
    keys, total = sorted(set(votes) | set(supports)), float(sum(votes.values()))
    if not keys or total <= 0:
        return {}
    logits = []
    for key in keys:
        share = (votes.get(key, 0) + VOTE_SMOOTH) / (total + VOTE_SMOOTH * len(keys))
        logits.append(math.log(share) + EVIDENCE_WEIGHT * float(supports.get(key, 0.0)))
    temperature = 1.0 + JUDGE_WEIGHT * abs(2.0 * s_bar_o(judge_score) - 1.0)
    top = max(logits)
    weights = [math.exp(temperature * (logit - top)) for logit in logits]
    normaliser = sum(weights)
    return {key: weight / normaliser for key, weight in zip(keys, weights)}


def state_distribution(state: Any, dataset_name: str) -> Dict[str, float]:
    """Score one finished question's clusters: agent votes, verdict, support, Judge."""
    yes_no = dataset_name == "StrategyQA"
    answers = list(agent_answers(state, yes_no))
    final = state.pred or state.raw_pred
    answers += [final] if final else []
    votes: Dict[str, int] = {}
    representative: Dict[str, str] = {}
    for answer in answers:
        key = cluster_key(answer, yes_no)
        if not key:
            continue
        votes[key] = votes.get(key, 0) + 1
        representative.setdefault(key, answer)
    passages = passage_texts(state.query_pool)
    supports = {key: evidence_support(text, passages) for key, text in representative.items()}
    return probability_distribution(votes, supports, getattr(state, "judge_score", None))


def _check_alpha(alpha: float) -> None:
    valid = isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and \
        0.0 < float(alpha) < 1.0
    if not valid:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile, capped at 1.

    A rank above ``n`` is too small for this alpha; bounded scores then give 1."""
    _check_alpha(alpha)
    values = sorted(float(score) for score in scores)
    rank = math.ceil((len(values) + 1) * (1.0 - float(alpha)))
    if rank > len(values):
        return 1.0
    return values[rank - 1]


def calibrate(examples: Iterable[Tuple[str, str, Mapping[str, float]]], alpha: float,
              min_n: int = 20, split: str = "dev") -> Dict[str, Any]:
    """Build a calibration document from ``(dataset, gold_cluster, distribution)`` rows."""
    _check_alpha(alpha)
    if min_n < 1:
        raise ValueError("min_n must be at least 1")
    if split == "test":
        raise CalibrationError("refusing to calibrate on the test split")
    per_dataset: Dict[str, list] = {}
    pooled = []
    for dataset, gold, distribution in examples:
        if not dataset:
            raise CalibrationError("every calibration row needs a dataset")
        score = nonconformity(distribution.get(gold, 0.0))
        per_dataset.setdefault(dataset, []).append(score)
        pooled.append(score)
    if not pooled:
        raise CalibrationError("calibration needs at least one row")
    return {
        "format_version": FORMAT_VERSION, "alpha": float(alpha), "split": split,
        "min_n": int(min_n),
        "pooled": {"n": len(pooled), "qhat": conformal_quantile(pooled, alpha)},
        "datasets": {name: {"n": len(scores), "qhat": conformal_quantile(scores, alpha)}
                     for name, scores in sorted(per_dataset.items())},
    }


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CalibrationError(f"{name} must be in [0, 1], got {value!r}")
    return number


def _dataset_entry(name: str, entry: Any) -> Tuple[int, float]:
    if not isinstance(entry, dict):
        raise CalibrationError(f"dataset {name!r} must be an object")
    count = entry.get("n")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CalibrationError(f"dataset {name!r} needs a non-negative integer n")
    return count, _unit_interval(entry.get("qhat"), f"dataset {name!r} qhat")


def _header(data: dict, alpha: float) -> int:
    """Validate format, split, alpha and ``min_n``; return ``min_n``."""
    if data.get("format_version") != FORMAT_VERSION:
        raise CalibrationError(f"format_version must be {FORMAT_VERSION}")
    if data.get("split") == "test":
        raise CalibrationError("refusing a calibration file built on the test split")
    stored = _unit_interval(data.get("alpha"), "alpha")
    if not 0.0 < stored < 1.0:
        raise CalibrationError("alpha must be in (0, 1)")
    if abs(stored - float(alpha)) > 1e-12:
        raise CalibrationError(
            f"calibration alpha {stored} does not match conformal_alpha {alpha}")
    min_n = data.get("min_n")
    if isinstance(min_n, bool) or not isinstance(min_n, int) or min_n < 1:
        raise CalibrationError("min_n must be a positive integer")
    return min_n


def _thresholds(data: dict) -> Tuple[int, float, Dict[str, Tuple[int, float]]]:
    """Validate the pooled block and every per-dataset entry."""
    pooled = data.get("pooled")
    if not isinstance(pooled, dict):
        raise CalibrationError("pooled must be an object")
    pooled_n, pooled_qhat = _dataset_entry("pooled", pooled)
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise CalibrationError("datasets must be an object")
    entries = {name: _dataset_entry(name, entry) for name, entry in datasets.items()}
    return pooled_n, pooled_qhat, entries


def load_calibration(path: Any, alpha: float) -> Calibration:
    """Read and validate one calibration file; any mismatch raises ``CalibrationError``."""
    _check_alpha(alpha)
    if path is None or not str(path).strip():
        raise CalibrationError("conformal_alpha needs conformal_model pointing at a file")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration {str(path)!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"calibration {str(path)!r} is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError("calibration must be a JSON object")
    pooled_n, pooled_qhat, entries = _thresholds(data)
    return Calibration(float(alpha), _header(data, alpha), pooled_n, pooled_qhat,
                       entries, str(data.get("split", "")))


def select_threshold(calibration: Calibration,
                     dataset: str) -> Tuple[float, bool]:
    """The dataset threshold, or ``(pooled, True)`` when that dataset is too small."""
    entry = calibration.datasets.get(dataset)
    if entry is not None and entry[0] >= calibration.min_n:
        return entry[1], False
    return calibration.pooled_qhat, True


def prediction_set(distribution: Mapping[str, float], qhat: float) -> list:
    """Clusters with ``p_hat >= 1 - q_hat``; an empty result also means abstain."""
    threshold = 1.0 - min(1.0, max(0.0, float(qhat)))
    return sorted(key for key, probability in distribution.items()
                  if probability >= threshold)


def apply_conformal(ctx: Any, state: Any) -> None:
    """Final step: record the set and mark abstention/escalation.

    ``state.pred`` is untouched, so full-coverage EM stays comparable; selective
    metrics exclude rows whose ``conformal_abstained`` is set.
    """
    alpha = ctx.flags.conformal_alpha
    if alpha is None:
        return
    calibration = load_calibration(ctx.flags.conformal_model, alpha)
    qhat, pooled = select_threshold(calibration, ctx.dataset_name)
    answer_set = prediction_set(state_distribution(state, ctx.dataset_name), qhat)
    abstained = len(answer_set) != 1
    state.conformal_qhat = qhat
    state.conformal_set = list(answer_set)
    state.conformal_abstained = abstained
    state.conformal_escalated = abstained and ctx.flags.conformal_escalate
    state.emit("conformal_qhat", qhat)
    state.emit("conformal_set", list(answer_set))
    state.emit("conformal_abstained", abstained)
    if state.conformal_escalated:
        state.emit("conformal_escalated", True)
    if pooled:
        state.emit("conformal_pooled", True)
