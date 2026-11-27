"""Candidate bindings, grounding and the width-2 beam rule.

A candidate's confidence is the fraction of the hop's passages that contain the entity
as a whole normalised string. The default candidates are the generated answer's explicit
alternatives ("X or Y", "X; Y"); two survive only when their confidences differ by less
than the margin. A title is never invented as a binding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from ..analysis.extract import extract_answer

BEAM_WIDTH = 2
BEAM_MARGIN = 0.15
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Candidate:
    """One possible binding for a hop and the passage support that justifies it."""

    entity: str
    confidence: float


def _normalise(text: object) -> str:
    return _NON_WORD.sub(" ", str(text).lower()).strip()


def grounded(entity: str, passages: Sequence[dict]) -> bool:
    """Whole-string containment of the normalised entity in any passage."""
    needle = _normalise(entity)
    if not needle:
        return False
    return any(f" {needle} " in f" {_normalise(p.get('contents', ''))} "
               for p in passages)


def doc_ids(passages: Sequence[dict]) -> tuple:
    return tuple(str(passage.get("id", "")) for passage in passages
                 if passage.get("id") is not None)


def confidence(entity: str, passages: Sequence[dict]) -> float:
    if not passages:
        return 0.0
    hits = sum(1 for passage in passages if grounded(entity, [passage]))
    return hits / len(passages)


def default_candidates(question: str, answer: str, passages: Sequence[dict]
                       ) -> List[Candidate]:
    """Explicit alternatives in the generated answer, deduplicated and capped at two."""
    found: List[Candidate] = []
    for entity in re.split(r"\s+or\s+|;", extract_answer(answer, False)):
        text = str(entity).strip()
        if text and _normalise(text) not in {_normalise(c.entity) for c in found}:
            found.append(Candidate(text, confidence(text, passages)))
    return sorted(found, key=lambda c: (-c.confidence, _normalise(c.entity)))[:BEAM_WIDTH]


def select_candidates(candidates: Sequence[Candidate], margin: float = BEAM_MARGIN,
                      width: int = BEAM_WIDTH) -> List[Candidate]:
    """Keep the top two only when the runner-up is within ``margin``; otherwise one."""
    ranked = sorted(candidates, key=lambda c: (-c.confidence, _normalise(c.entity)))
    if len(ranked) < 2:
        return ranked[:1]
    if ranked[0].confidence - ranked[1].confidence < margin:
        return ranked[:width]
    return ranked[:1]
