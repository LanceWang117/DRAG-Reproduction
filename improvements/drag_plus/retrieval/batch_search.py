"""Single-query search through the batch path (B2 fix, J30).

At the pinned FlashRAG (0.1.4.dev20250111), ``DenseRetriever.search(query)`` passes a
bare string to the encoder. The encoder slices it as if it were a list of queries, so
only ``query[:64]`` (64 characters including E5's ``"query: "`` prefix) is embedded.
``batch_search([query])`` embeds the whole query; Naive already uses it, and the DRAG
query stage did not (findings B2: 32 of 124 unchanged pools differed from Naive's
retrieval, all with queries over 57 characters).

With ``search_via_batch`` on, ``search(q)`` is exactly ``batch_search([q])[0]``, with
``num`` and ``return_score`` passed through. The default (off) keeps the recorded
baseline path, so T0 and E01's baseline arms are unchanged.
"""
from __future__ import annotations

from typing import Any, List, Optional


class BatchSearchRetriever:
    """Wrap a retriever so one query takes the batch path; everything else delegates."""

    def __init__(self, inner: Any):
        self.inner = inner

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False) -> Any:
        if return_score:
            rows, scores = self.inner.batch_search([query], **_options(num, True))
            return rows[0], scores[0]
        return self.inner.batch_search([query], **_options(num, False))[0]

    def batch_search(self, queries: List[str], num: Optional[int] = None,
                     return_score: bool = False) -> Any:
        return self.inner.batch_search(queries, **_options(num, return_score))

    def __getattr__(self, name: str) -> Any:  # e.g. FlashRAG attributes the runner reads
        return getattr(self.inner, name)


def _options(num: Optional[int], return_score: bool) -> dict:
    """Pass only what was asked for, so retrievers without these parameters still work."""
    options = {}
    if num is not None:
        options["num"] = num
    if return_score:
        options["return_score"] = True
    return options


def with_batch_search(retriever: Any, flags: Any) -> Any:
    """The B2 fix when ``search_via_batch`` is on; the retriever unchanged otherwise."""
    return BatchSearchRetriever(retriever) if flags.search_via_batch else retriever
