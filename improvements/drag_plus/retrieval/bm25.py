"""BM25 adapter; ``bm25s`` is imported on first use, never when the module loads.

E04 builds the real index over the 21M-passage corpus. The index directory must be
saved with its corpus (``BM25.save(dir, corpus=...)``) as a list of ``{"id",
"contents"}`` dicts: ``bm25s`` retrieval is position-based and returns ``corpus[i]``,
so the order and length of the saved corpus define the passage ids.

Both the loader and the query tokeniser are injected, because the query tokenisation
must match whatever the index was built with (stemming included) and tests must run
without the library installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple


def load_bm25s(path: Path) -> Any:
    """Load a saved index memory-mapped; the import is the expensive part."""
    import bm25s

    return bm25s.BM25.load(str(path), load_corpus=True, mmap=True)


def tokenize_bm25s(queries: Sequence[str]) -> Any:
    """Query tokenisation with ``bm25s``'s own defaults (lowercase, stopwords, no stem).

    The index build must tokenise its corpus the same way or the query tokens miss the
    vocabulary. E04 owns that build; this call is the default its README implies, and an
    index built differently passes its own ``tokenize``.
    """
    import bm25s

    return bm25s.tokenize(list(queries), show_progress=False)


class Bm25Retriever:
    """The lexical retriever: ``search`` and ``batch_search`` over a saved index."""

    def __init__(self, path: Any, *, depth: int = 20,
                 load: Optional[Callable[[Path], Any]] = None,
                 tokenize: Optional[Callable[[Sequence[str]], Any]] = None) -> None:
        self.path = Path(path)
        self.depth = depth
        self._load = load or load_bm25s
        self._tokenize = tokenize or tokenize_bm25s
        self._index: Optional[Any] = None

    @property
    def index(self) -> Any:
        """Load once, on the first query, so an unused BM25 arm costs nothing."""
        if self._index is None:
            if not self.path.exists():
                raise FileNotFoundError(f"BM25 index not found: {self.path}")
            self._index = self._load(self.path)
        return self._index

    def search(self, query: str, num: Optional[int] = None,
               return_score: bool = False) -> Any:
        """One query as the batch path with a single-element request."""
        result = self.batch_search([query], num=num, return_score=return_score)
        if return_score:
            return result[0][0], result[1][0]
        return result[0]

    def batch_search(self, queries: List[str], num: Optional[int] = None,
                     return_score: bool = False) -> Any:
        """Every query in one ``retrieve`` call; ``show_progress`` stays off for logs."""
        queries = [queries] if isinstance(queries, str) else list(queries)
        depth = self.depth if num is None else int(num)
        if depth < 1:
            raise ValueError(f"BM25 depth must be >= 1, got {depth}")
        results = self.index.retrieve(self._tokenize(queries), k=depth,
                                      show_progress=False)
        documents = [self._documents(row) for row in results.documents]
        if not return_score:
            return documents
        return documents, [[float(score) for score in row] for row in results.scores]

    @staticmethod
    def _documents(row: Any) -> List[dict]:
        """One result row as fresh dicts; the index must hold a dict corpus."""
        documents = []
        for entry in row:
            if not isinstance(entry, dict) or "contents" not in entry:
                raise ValueError("the BM25 index must be saved with a corpus of "
                                 "{'id', 'contents'} dicts")
            documents.append(dict(entry))
        return documents
