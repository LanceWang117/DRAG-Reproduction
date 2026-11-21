"""Normalise a proposed query so near-identical rewrites compare equal.

The rule is deliberately lexical (J07): Unicode NFC, lowercase, punctuation to spaces,
whitespace collapse, then drop a fixed list of articles, auxiliaries and two
prepositions. Equal token *sets* mean "effectively the same query". The list is smaller
than Lucene's English stop set: it keeps ``no``, ``not`` and ``to``, which can flip an
answer, and adds ``were``/``been`` from the Snowball list.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable

STOP_WORDS = frozenset({"a", "an", "the", "is", "was", "are", "were", "be", "been",
                        "of", "for"})


def normalize_query(text: str) -> frozenset:
    """Return the comparison key of ``text``: its non-stop-word token set.

    Punctuation becomes a space, so ``shiva—or—siva`` splits into words. All-stop-word
    and empty input yield the empty set, which is a valid key: two such queries are
    duplicates of each other.
    """
    normalized = unicodedata.normalize("NFC", str(text)).lower()
    words = "".join(" " if unicodedata.category(ch).startswith("P") else ch
                    for ch in normalized).split()
    return frozenset(word for word in words if word not in STOP_WORDS)


def is_duplicate_query(query: str, existing: Iterable[str]) -> bool:
    """True when ``query`` has the same normalised key as any key in ``existing``."""
    key = normalize_query(query)
    return any(normalize_query(other) == key for other in existing)
