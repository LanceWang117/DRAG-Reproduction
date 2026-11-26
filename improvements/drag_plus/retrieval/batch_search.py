"""Single-query search through the batch path (B2 fix, J30).

At the pinned FlashRAG (0.1.4.dev20250111), ``DenseRetriever.search(query)`` passes a
bare string to the encoder. The encoder slices it as if it were a list of queries, so
only ``query[:64]`` (64 characters including E5's ``"query: "`` prefix) is embedded.
``batch_search([query])`` embeds the whole query; Naive already uses it, and the DRAG
query stage did not (findings B2: 32 of 124 unchanged pools differed from Naive's
retrieval, all with queries over 57 characters).

With ``search_via_batch`` on, ``search(q)`` is exactly ``batch_search([q])[0]``, with
``num`` and ``return_score`` passed through. The default (off) keeps the recorded
baseline path, so T0 and E01's baseline arms are unchanged. The retrieval cache already
takes the batch path, so the runner applies this wrapper only without it.

With the flag off and ``question_batch_size`` > 1, a search grouped with others goes
through ``batch_search`` (full query) while a lone one goes through ``search``
(truncated). That mix predates J30 (J15/J16 grouping); every generated arm that groups
questions sets this flag, so no arm mixes the two paths.
"""
from __future__ import annotations

from typing import Any, Optional


class BatchSearchRetriever:
    """Wrap a retriever so one query takes the batch path; everything else delegates."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.batch_search = inner.batch_search  # the real method, so signature checks see it

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False) -> Any:
        if return_score:
            rows, scores = self.inner.batch_search([query], **_options(num, True))
            return rows[0], scores[0]
        return self.inner.batch_search([query], **_options(num, False))[0]

    def __getattr__(self, name: str) -> Any:  # e.g. FlashRAG attributes the runner reads
        if name == "inner":  # not set yet (copy, pickle): never recurse
            raise AttributeError(name)
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
    """The B2 fix when ``search_via_batch`` is on and no cache already provides it."""
    if flags.search_via_batch and flags.retrieval_cache is None:
        return BatchSearchRetriever(retriever)
    return retriever
