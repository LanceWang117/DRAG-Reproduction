"""The two answer-extraction policies, per arm.

``native`` is what the original code scored: DRAG's raw moderator text, and the Naive arm's
text after ``single_agent_pred_parse``. ``unified`` is E(.) from the analysis package,
applied identically to every arm (finding B12).
"""
from __future__ import annotations

from .analysis.extract import extract_answer

FINAL_ANSWER_PREFIX = "The answer is:"


def parse_naive(pred: str) -> str:
    """Same rule as ``single_agent_pred_parse``: the segment after the first marker."""
    if FINAL_ANSWER_PREFIX in pred:
        return pred.split(FINAL_ANSWER_PREFIX)[1].strip()
    return pred


def native_pred(arm: str, raw: str) -> str:
    return parse_naive(raw) if arm == "naive" else raw


def unified_pred(raw: str, dataset_name: str) -> str:
    return extract_answer(raw, yes_no=dataset_name == "StrategyQA")
