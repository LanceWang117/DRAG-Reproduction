"""Faithfulness scoring and the J19 weighted aggregation rule.

A faithfulness scorer maps ``(answer, passages)`` to a soft score in ``[0, 1]``. The
default ``heuristic`` implementation is deterministic and needs no model: it normalises
the answer with E(.) and returns the fraction of cited passages that contain it as a
whole word. A yes/no answer returns ``0.5`` ("unknown") because both words occur in
almost every passage, so a literal match would always look perfect. ``minicheck`` is
an optional adapter imported only when configured; a missing package is a configuration
error, never a silent fallback.

The module also owns the linear weighted aggregation used by ``aggregator="weighted"``:
``score(a) = w_conf*conf(a) + w_faith*s_faith(a) + w_agr*agr(a)``. ``conf`` is the
answer's plurality share, ``s_faith`` the scorer value, and ``agr`` the mean token-F1
against the other distinct answers. Weights come from a JSON file that records the
training split and the SHA-256 of its input manifest; the loader refuses ``test``.

Edge cases: an empty answer or no passages scores 0.0; an unknown scorer name, a
missing/malformed weights file, a test split, a non-finite score, or a non-positive
total weight is a ``FaithfulnessConfigError``. Aggregation returns ``("", "empty", [])``
when every candidate is empty and breaks exact score ties by first-seen order.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..analysis.extract import extract_answer

SIGNAL_NAMES: Tuple[str, str, str] = ("conf", "s_faith", "agr")
_YES_NO = frozenset(("yes", "no"))
_WORD = re.compile(r"[a-z0-9]+")
_HEX = re.compile(r"[0-9a-f]{64}")


class FaithfulnessConfigError(ValueError):
    """A scorer or weights configuration is missing, malformed or test-split."""


class FaithfulnessScorer(Protocol):
    """One soft support score for an answer against the cited passages."""

    def __call__(self, answer: str, passages: Sequence[str]) -> float: ...


def _normalise_text(text: Any) -> str:
    return " ".join(str(text).casefold().split())


def _contains(passage: str, answer: str) -> bool:
    """Whole-word containment; ``Paris`` does not match ``Parisian``."""
    pattern = r"(?<!\w)" + re.escape(answer.casefold()) + r"(?!\w)"
    return re.search(pattern, passage) is not None


class HeuristicFaithfulness:
    """Deterministic whole-word containment, with yes/no treated as unknown."""

    def __init__(self, yes_no: bool = False):
        self.yes_no = yes_no

    def score(self, answer: str, passages: Sequence[str]) -> float:
        """Fraction of non-empty passages containing the E(.)-normalised answer."""
        normalised = _normalise_text(extract_answer(answer, yes_no=self.yes_no))
        if not normalised:
            return 0.0
        if normalised in _YES_NO:
            return 0.5
        texts = [_normalise_text(passage) for passage in passages
                 if str(passage).strip()]
        if not texts:
            return 0.0
        return sum(_contains(text, normalised) for text in texts) / len(texts)

    def __call__(self, answer: str, passages: Sequence[str]) -> float:
        return self.score(answer, passages)


def build_scorer(name: Optional[str], yes_no: bool = False) -> FaithfulnessScorer:
    """Return the configured scorer; ``None`` and ``heuristic`` are the default."""
    if name in (None, "heuristic"):
        return HeuristicFaithfulness(yes_no)
    if name == "minicheck":
        from .minicheck import MiniCheckFaithfulness
        return MiniCheckFaithfulness(yes_no=yes_no)
    raise FaithfulnessConfigError(
        f"unknown faithfulness_scorer {name!r}; expected 'heuristic' or 'minicheck'")


@dataclass(frozen=True)
class AggregatorWeights:
    """The three non-negative linear weights plus their provenance."""

    conf: float
    s_faith: float
    agr: float
    training_split: str
    manifest_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {"weights": {name: getattr(self, name) for name in SIGNAL_NAMES},
                "training_split": self.training_split,
                "manifest_sha256": self.manifest_sha256}


def _finite_weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FaithfulnessConfigError(f"weight {name!r} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise FaithfulnessConfigError(f"weight {name!r} must be finite and >= 0")
    return number


def _read_weights(path: Any) -> Mapping[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise FaithfulnessConfigError(
            f"cannot read aggregator weights {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise FaithfulnessConfigError("aggregator weights must hold a JSON object")
    return data


def load_weights(path: Any) -> AggregatorWeights:
    """Load and validate a weights JSON; a ``test`` split is refused."""
    data = _read_weights(path)
    split = data.get("training_split")
    if not isinstance(split, str) or not split.strip():
        raise FaithfulnessConfigError("aggregator weights need a non-empty training_split")
    if split.casefold() == "test":
        raise FaithfulnessConfigError("refusing aggregator weights trained on the test split")
    digest = data.get("manifest_sha256")
    if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
        raise FaithfulnessConfigError("aggregator weights need a sha256 manifest_sha256")
    weights = data.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(SIGNAL_NAMES):
        raise FaithfulnessConfigError(
            f"aggregator weights must have exactly {list(SIGNAL_NAMES)}")
    values = {name: _finite_weight(weights[name], name) for name in SIGNAL_NAMES}
    if sum(values.values()) <= 0:
        raise FaithfulnessConfigError("aggregator weights must have a positive total")
    return AggregatorWeights(training_split=split, manifest_sha256=digest, **values)


def _tokens(answer: str) -> set:
    return set(_WORD.findall(answer.casefold()))


def _token_f1(left: str, right: str) -> float:
    first, second = _tokens(left), _tokens(right)
    if not first or not second:
        return 0.0
    overlap = len(first & second)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(first), overlap / len(second)
    return 2 * precision * recall / (precision + recall)


def agreement(answer: str, others: Sequence[str]) -> float:
    """Mean token-F1 to the other distinct answers; alone, agreement is 1.0."""
    if not others:
        return 1.0
    return sum(_token_f1(answer, other) for other in others) / len(others)


def _group_answers(answers: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for source, raw in answers:
        answer = str(raw or "").strip()
        if answer:
            groups.setdefault(answer, []).append(source)
    return groups


def _score(row: Mapping[str, Any], weights: AggregatorWeights) -> float:
    return (weights.conf * row["conf"] + weights.s_faith * row["s_faith"]
            + weights.agr * row["agr"])


def select_weighted(answers: Sequence[Tuple[str, str]], passages: Sequence[str],
                    scorer: FaithfulnessScorer, weights: AggregatorWeights
                    ) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Pick the highest weighted candidate; ties keep the first-seen answer."""
    groups = _group_answers(answers)
    if not groups:
        return "", "empty", []
    total = sum(len(sources) for sources in groups.values())
    distinct = list(groups)
    rows = []
    for answer, sources in groups.items():
        faith = float(scorer(answer, passages))
        if not math.isfinite(faith):
            raise FaithfulnessConfigError("faithfulness scorer returned a non-finite value")
        row = {"answer": answer, "sources": list(sources),
               "conf": len(sources) / total,
               "s_faith": min(max(faith, 0.0), 1.0),
               "agr": agreement(answer, [other for other in distinct if other != answer])}
        row["score"] = _score(row, weights)
        rows.append(row)
    best = max(rows, key=lambda row: row["score"])
    tied = [row for row in rows if row["score"] == best["score"]]
    return best["answer"], "weighted" if len(tied) == 1 else "first_seen", rows
