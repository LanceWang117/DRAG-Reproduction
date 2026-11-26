"""Pre-debate features for the J20 cost-aware router.

The router must decide a route before any debate round runs. This module therefore reads
only information available at that point:

* ``retrieval_margin`` - top-1 minus top-2 retrieval score from the original retrieval;
  ``0.0`` when fewer than two finite scores exist.
* ``s_bar_O`` - ``sigmoid`` of the J12 Judge margin, ``0.5`` when the probe is unavailable.
* ``answer_agreement`` - ``1.0`` when the model's own answer and the retrieval-based
  extractive answer agree by normalised token containment, otherwise ``0.0``.
* ``naive_entropy`` - normalised Shannon entropy over the Naive answer's tokens; a cheap
  pre-debate proxy for answer uncertainty, ``0.0`` for fewer than two distinct tokens.

The leakage guard is structural: ``RouterInputs`` has exactly four fields, none of which
is a gold answer, a label, or an output produced after the router decision. ``FEATURE_INPUTS``
names that whitelist and the tests assert it never grows silently.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .features import s_bar_o

ROUTER_FEATURE_NAMES: Tuple[str, ...] = (
    "retrieval_margin", "s_bar_O", "answer_agreement", "naive_entropy")
FEATURE_INPUTS: Tuple[str, ...] = (
    "passages", "judge_score", "own_answer", "retrieval_answer")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RouterInputs:
    """The only data the router features may read, in a fixed, leak-free shape."""

    passages: Sequence[Mapping] = ()
    judge_score: Optional[float] = None
    own_answer: str = ""
    retrieval_answer: str = ""


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _tokens(text: str) -> List[str]:
    return _WORD.findall(str(text).lower())


def _scores(passages: Iterable[Mapping]) -> List[float]:
    values = []
    for passage in passages:
        score = _finite(passage.get("score")) if isinstance(passage, Mapping) else None
        if score is not None:
            values.append(score)
    return values


def retrieval_margin(passages: Iterable[Mapping]) -> float:
    """Top-1 minus top-2 score; unsorted input is sorted first, missing scores are skipped."""
    values = sorted(_scores(passages), reverse=True)
    return float(values[0] - values[1]) if len(values) >= 2 else 0.0


def naive_entropy(answer: str) -> float:
    """Normalised token entropy in ``[0, 1]``; a one-token or empty answer is zero."""
    tokens = _tokens(answer)
    if len(set(tokens)) < 2:
        return 0.0
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def answer_agreement(own_answer: str, retrieval_answer: str) -> float:
    """Whether two answers agree: equal token sets, or one a non-empty subset of the other."""
    own, retrieval = set(_tokens(own_answer)), set(_tokens(retrieval_answer))
    if not own or not retrieval:
        return 0.0
    return 1.0 if own == retrieval or own < retrieval or retrieval < own else 0.0


def extractive_answer(passages: Iterable[Mapping]) -> str:
    """The retrieval-based answer proxy: the first line (title) of the top passage."""
    for passage in passages:
        if not isinstance(passage, Mapping):
            continue
        content = str(passage.get("contents", ""))
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            return lines[0]
    return ""


def top_passages(query_pool: Mapping[str, list]) -> List[Mapping]:
    """The original query's passages, or every passage when the pool is empty."""
    if query_pool:
        return list(next(iter(query_pool.values())))
    return []


def inputs_from_state(state, own_answer: str) -> RouterInputs:
    """Build the four whitelisted inputs from a question state and the Naive answer."""
    passages = top_passages(state.query_pool)
    return RouterInputs(passages=passages, judge_score=getattr(state, "judge_score", None),
                        own_answer=own_answer,
                        retrieval_answer=extractive_answer(passages))


def compute_features(inputs: RouterInputs) -> Mapping[str, float]:
    """Return every router feature, in ``ROUTER_FEATURE_NAMES`` order, all finite."""
    values = {
        "retrieval_margin": retrieval_margin(inputs.passages),
        "s_bar_O": s_bar_o(inputs.judge_score),
        "answer_agreement": answer_agreement(inputs.own_answer, inputs.retrieval_answer),
        "naive_entropy": naive_entropy(inputs.own_answer),
    }
    return {name: float(values[name]) for name in ROUTER_FEATURE_NAMES}
