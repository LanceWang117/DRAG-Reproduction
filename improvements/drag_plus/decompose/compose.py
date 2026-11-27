"""Final composition for a grounded decomposition.

The ledger is the route's external memory: one ``(sub_q, entity, confidence, doc_ids)``
row per completed hop. Composition refuses a ledger that is shorter than the plan, out
of order, ungrounded (no doc id) or empty, so an ungrounded binding can never become the
final answer. The text ends with ``The answer is: <final>`` so the unified extractor sees
the same marker as the baseline moderator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .plan import Plan

FINAL_MARKER = "The answer is:"


class CompositionError(ValueError):
    """The ledger cannot support the requested final answer."""


@dataclass(frozen=True)
class LedgerEntry:
    """One external ledger row: the hop's bound entity and the passages that support it."""

    sub_q: int
    entity: str
    confidence: float
    doc_ids: Tuple[str, ...]

    def as_dict(self) -> dict:
        return {"sub_q": self.sub_q, "entity": self.entity,
                "confidence": self.confidence, "doc_ids": list(self.doc_ids)}


@dataclass(frozen=True)
class Composition:
    """The final answer, its answer-bearing text and every supporting document id."""

    answer: str
    text: str
    doc_ids: Tuple[str, ...]


def _check_ledger(plan: Plan, ledger: Tuple[LedgerEntry, ...]) -> None:
    if len(ledger) != len(plan.steps):
        raise CompositionError(
            f"ledger has {len(ledger)} rows for {len(plan.steps)} hops")
    for step, entry in zip(plan.steps, ledger):
        if entry.sub_q != step.qid:
            raise CompositionError(f"ledger row {entry.sub_q} is out of order")
        if not isinstance(entry.entity, str) or not entry.entity.strip():
            raise CompositionError(f"hop {entry.sub_q} has an empty entity")
        if not entry.doc_ids:
            raise CompositionError(f"hop {entry.sub_q} has no supporting document id")
        if not 0.0 <= float(entry.confidence) <= 1.0:
            raise CompositionError(f"hop {entry.sub_q} confidence is outside [0, 1]")


def compose_answer(plan: Plan, ledger: Tuple[LedgerEntry, ...],
                   final_answer: str) -> Composition:
    """Validate the ledger and build the composed, answer-bearing text."""
    _check_ledger(plan, ledger)
    answer = str(final_answer).strip()
    if not answer:
        raise CompositionError("the final answer is empty")
    chain = "; ".join(f"q{entry.sub_q}: {entry.entity.strip()} "
                      f"[{', '.join(entry.doc_ids)}]" for entry in ledger)
    text = f"Sub-question answers: {chain}.\n{FINAL_MARKER} {answer}."
    doc_ids = tuple(dict.fromkeys(doc for entry in ledger for doc in entry.doc_ids))
    return Composition(answer, text, doc_ids)
