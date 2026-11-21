"""Append-only evidence store keyed by ``doc_id``.

The original question's passages are anchored: an Optimization or Expansion can add
passages and supersede later queries, but it cannot evict an anchored passage. All
iteration is in first-seen order (round, then rank), which is the prefix-cache
precondition for J11.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .format import EvidenceEntry, format_entries, format_pool, label


class EvidenceStoreError(ValueError):
    """A passage cannot enter the store, usually because it has no doc id."""


@dataclass
class QueryRecord:
    """One query ever added to the store and the passages it introduced."""
    qid: int
    query: str
    passages: List[dict] = field(default_factory=list)
    anchored: bool = False
    superseded: bool = False


class AnchoredEvidenceStore:
    """Append-only, deterministic evidence for one question."""

    def __init__(self) -> None:
        self.entries: List[EvidenceEntry] = []
        self._by_id: Dict[str, EvidenceEntry] = {}
        self._labels: set = set()
        self._queries: Dict[str, QueryRecord] = {}
        self._next_qid = 1

    def add_query(self, query: str, passages: List[dict], round_index: int,
                  anchored: bool = False) -> int:
        """Add one retrieval result; duplicate doc ids are skipped, never relabelled."""
        record = self._queries.get(query)
        if record is None:
            record = QueryRecord(self._next_qid, query, anchored=anchored)
            self._next_qid += 1
            self._queries[query] = record
        elif anchored:
            record.anchored = True
        for position, passage in enumerate(passages, start=1):
            doc_id = self._doc_id(passage)
            if doc_id in self._by_id:
                continue
            rank = self._free_rank(record.qid, position)
            entry = EvidenceEntry(doc_id, str(passage.get("contents", "")),
                                  label(record.qid, rank), record.qid, rank,
                                  round_index, record.anchored)
            self.entries.append(entry)
            self._by_id[doc_id] = entry
            self._labels.add(entry.label)
            record.passages.append(passage)
        return record.qid

    def _doc_id(self, passage: dict) -> str:
        doc_id = str(passage.get("id", "") or "")
        if not doc_id:
            raise EvidenceStoreError(f"passage has no doc id: {passage!r}")
        return doc_id

    def _free_rank(self, qid: int, start: int) -> int:
        """First rank for this query whose label is not already used."""
        rank = start
        while label(qid, rank) in self._labels:
            rank += 1
        return rank

    def supersede(self, query: str) -> bool:
        """Mark a non-original query superseded; its passages stay in the store."""
        record = self._queries.get(query)
        if record is None or record.anchored:
            return False
        record.superseded = True
        return True

    def query_record(self, query: str) -> Optional[QueryRecord]:
        return self._queries.get(query)

    def ids(self) -> set:
        """Every doc id ever admitted, including superseded queries' passages."""
        return set(self._by_id)

    def active_pool(self) -> Dict[str, List[dict]]:
        """Non-superseded queries; anchored queries never become superseded."""
        return {query: list(record.passages) for query, record in self._queries.items()
                if not record.superseded}

    def to_query_pool(self) -> Dict[str, List[dict]]:
        """Every query ever added, for FlashRAG's saved ``QueryStage_QueryPool`` key."""
        return {query: list(record.passages) for query, record in self._queries.items()}

    def format(self, delimiters: bool = False) -> str:
        return format_entries(self.entries, delimiters)


class EvidencePool(dict):
    """A query pool that can render itself with global evidence labels.

    With ``entries`` given (anchored mode) the store owns the rendering, including
    superseded queries' surviving passages. Without it (only delimiters on) the labels
    are derived from the current baseline pool at render time.
    """

    def __init__(self, *args, entries=None, delimiters: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = entries
        self.delimiters = delimiters

    def format_evidence(self) -> str:
        if self.entries is not None:
            return format_entries(self.entries, self.delimiters)
        return format_pool(self, self.delimiters)
