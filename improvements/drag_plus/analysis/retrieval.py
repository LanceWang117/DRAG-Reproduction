"""Retrieval-only metrics computed from candidate lists alone (J17).

These functions never retrieve anything and never load a model, so the same code scores
a fresh run dump and the passages already stored in ``results/retrieval-200.jsonl``.

Relevance is binary: a passage is relevant when a normalised gold answer occurs in it
(J01's ``normalize_answer``, the scorer that reproduces FlashRAG). ``containment@k`` is
measured on the context the generator actually sees -- the question followed by the top-k
passages -- because the baseline prompt always contains the question. That is the
definition B17 used when it recorded 114/200 per arm; passages alone give 105/200, so the
two must never be mixed.
"""
from __future__ import annotations

from math import ceil, log2
from typing import Any, Dict, Iterable, List, Sequence

from .score import normalize_answer


def passage_title(contents: Any) -> str:
    """The baseline's title convention: the first line of ``contents``."""
    return str(contents).split("\n", 1)[0]


def contains_gold(text: str, golden_answers: Sequence[str]) -> bool:
    """True when some gold answer, normalised, occurs in ``text``."""
    haystack = normalize_answer(text)
    return any(normalize_answer(gold) in haystack for gold in golden_answers)


def relevance(passages: Sequence[dict], golden_answers: Sequence[str]) -> List[bool]:
    """One binary judgement per passage, in the given order."""
    return [contains_gold(str(passage.get("contents", "")), golden_answers)
            for passage in passages]


def containment_at_k(question: str, passages: Sequence[dict],
                     golden_answers: Sequence[str], k: int) -> float:
    """Gold in the context (question + top-k passages); see the module docstring.

    Each component is checked on its own: joining passages would let a gold answer match
    across a boundary and count as contained when no single passage holds it.
    """
    if contains_gold(question, golden_answers):
        return 1.0
    return float(any(contains_gold(str(passage.get("contents", "")), golden_answers)
                     for passage in list(passages)[:k]))


def reciprocal_rank(judged: Sequence[bool]) -> float:
    """1 over the rank of the first relevant passage, 0 when nothing is relevant."""
    for rank, relevant in enumerate(judged, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(judged: Sequence[bool], k: int) -> float:
    """Binary-gain nDCG@k; the ideal ranking puts every relevant passage first."""
    gains = [1.0 if relevant else 0.0 for relevant in list(judged)[:k]]
    dcg = sum(gain / log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1.0 / log2(index + 2) for index in range(int(sum(gains))))
    return dcg / ideal if ideal else 0.0


def unique_title_ratio(passages: Sequence[dict]) -> float:
    """Distinct article titles over passages: 1.0 means no two passages share one."""
    if not passages:
        return 0.0
    titles = {passage_title(passage.get("contents", "")) for passage in passages}
    return len(titles) / len(passages)


def evaluate(question: str, golden_answers: Sequence[str], passages: Sequence[dict],
             ks: Sequence[int] = (3, 5), ndcg_k: int = 5) -> Dict[str, float]:
    """Every metric for one question, as a flat name -> value mapping.

    ``containment`` uses the whole pool, which is the number the ledger recorded for the
    stored baseline; ``containment@k`` truncates it to the top k.
    """
    judged = relevance(passages, golden_answers)
    scores = {"containment": containment_at_k(question, passages, golden_answers,
                                              len(passages))}
    scores.update({f"containment@{k}": containment_at_k(question, passages,
                                                        golden_answers, k)
                   for k in ks})
    scores["mrr"] = reciprocal_rank(judged)
    scores[f"ndcg@{ndcg_k}"] = ndcg_at_k(judged, ndcg_k)
    scores["unique_title_ratio"] = unique_title_ratio(passages)
    return scores


def summarise(rows: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Means over questions, plus ``n`` and the integer count behind each containment."""
    if not rows:
        return {"n": 0}
    keys = sorted(rows[0])
    summary: Dict[str, Any] = {key: sum(row[key] for row in rows) / len(rows)
                               for key in keys}
    summary["n"] = len(rows)
    for key in keys:
        if key.startswith("containment"):
            summary[f"{key}_count"] = int(round(summary[key] * len(rows)))
    return summary


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, so no interpolation is invented between samples."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def latency_summary(values: Iterable[float]) -> Dict[str, Any]:
    """p50 and p99 in milliseconds, rounded, with the sample count."""
    samples = list(values)
    if not samples:
        return {}
    return {"count": len(samples),
            "p50_ms": round(percentile(samples, 0.50), 3),
            "p99_ms": round(percentile(samples, 0.99), 3)}
