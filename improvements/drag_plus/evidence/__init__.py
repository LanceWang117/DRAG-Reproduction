"""Evidence stores, prompt formatting and rewrite auditing (J10).

The default store is the baseline ``dict`` of query -> passages. ``anchored`` keeps an
append-only, ``doc_id``-keyed view so an Optimization cannot evict the original
question's evidence; ``evidence_delimiters`` renders that view as labelled data blocks.
"""
from __future__ import annotations

from .audit import audit_rewrite, extract_entities, extract_numbers
from .format import EvidenceEntry, format_entries, format_pool, label
from .store import (AnchoredEvidenceStore, EvidencePool, EvidenceStoreError,
                    QueryRecord)

__all__ = [
    "AnchoredEvidenceStore",
    "EvidenceEntry",
    "EvidencePool",
    "EvidenceStoreError",
    "QueryRecord",
    "audit_rewrite",
    "extract_entities",
    "extract_numbers",
    "format_entries",
    "format_pool",
    "label",
]
