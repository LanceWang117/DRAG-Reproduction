"""Rules that span several flags (checked by ``Flags`` after each flag on its own).

Each rule is ``(the resolved values are invalid, why)``. A rule exists so that a
combination that would silently run something other than what the file says fails at
load time instead.
"""
from __future__ import annotations

from typing import Any, Mapping

GROUPING = ("question_batch_size > 1 requires search_via_batch=true (or retrieval_cache, or "
            "retriever='hybrid', which already search through batch_search): grouped "
            "searches embed the whole query, so every search must (B2)")


def batch_path_only(values: Mapping[str, Any]) -> bool:
    """Every DRAG search already takes batch_search: the J30 fix, the cache, or hybrid."""
    return bool(values["search_via_batch"]) or values["retrieval_cache"] is not None \
        or values["retriever"] == "hybrid"


def check_group_size(size: int, flags: Any) -> None:
    """The same rule for an explicit ``run_batch(batch_size=...)`` override."""
    if size < 1:
        raise ValueError("question_batch_size must be at least 1")
    names = ("search_via_batch", "retrieval_cache", "retriever")
    if size > 1 and not batch_path_only({name: getattr(flags, name) for name in names}):
        raise ValueError(GROUPING)

COMBINATIONS = (
    (lambda v: v["judge_order_balance"] and v["judge_scoring"] != "logprob",
     "judge_order_balance requires judge_scoring='logprob'"),
    (lambda v: v["escalation_gate"] == "risk" and (
        v["escalation_model"] is None or v["escalation_tau"] is None),
     "escalation_gate='risk' requires escalation_model and escalation_tau"),
    (lambda v: v["lazy_query_opponent"] and v["escalation_tau"] is None,
     "lazy_query_opponent requires escalation_tau"),
    (lambda v: v["aggregator"] == "weighted" and v["aggregator_weights"] is None,
     "aggregator='weighted' requires aggregator_weights"),
    # J31: grouped searches go through batch_search (whole query) and a lone search
    # through FlashRAG's search (first 64 characters, B2), so grouping would mix both.
    (lambda v: v["question_batch_size"] > 1 and not batch_path_only(v), GROUPING),
    (lambda v: v["conformal_alpha"] is not None
     and not 0.0 < v["conformal_alpha"] < 1.0,
     "conformal_alpha must be in (0, 1)"),
    (lambda v: v["conformal_alpha"] is not None and v["conformal_model"] is None,
     "conformal_alpha requires conformal_model"),
)
