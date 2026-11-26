"""Rules that span several flags (checked by ``Flags`` after each flag on its own).

Each rule is ``(the resolved values are invalid, why)``. A rule exists so that a
combination that would silently run something other than what the file says fails at
load time instead.
"""
from __future__ import annotations

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
    (lambda v: v["question_batch_size"] > 1 and not v["search_via_batch"],
     "question_batch_size > 1 requires search_via_batch=true: grouped searches embed the "
     "whole query, so every search must (B2)"),
)
