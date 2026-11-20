"""One retrieval path for both arms: a caching proxy over the injected retriever (J04).

``search`` is defined as ``batch_search([query])[0]``, so the single and batch entry
points can never disagree. On a miss, one ``batch_search`` resolves every query the
request is missing, results are deep-copied on the way out, and one JSONL line per query
is appended only after that call succeeded.

The key is ``(index, encoder, top_k, normalised query)``, so a different index or depth
cannot hit a stale entry. Queries are compared after Unicode NFC and whitespace
normalisation and are never lowercased: the key has to stay correct if a later job swaps
in a cased encoder, even though E5-base-v2 itself is uncased.
"""
from __future__ import annotations

import copy
import inspect
import json
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..instrument.logs import append_line


def normalise_query(query: str) -> str:
    """NFC plus collapsed whitespace; case is preserved on purpose."""
    if not isinstance(query, str):
        raise TypeError(f"a query must be a string, got {type(query).__name__}")
    return " ".join(unicodedata.normalize("NFC", query).split())


def index_identity(path: Any) -> str:
    """A cheap identifier for an index file: absolute path, byte size and mtime."""
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return str(resolved)
    return f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"


@dataclass(frozen=True)
class CacheKey:
    """Everything that must match for a hit to be valid."""

    index: str
    encoder: str
    top_k: Optional[int]
    query: str

    @classmethod
    def build(cls, identity: Mapping[str, Any], query: str,
              num: Optional[int] = None) -> "CacheKey":
        depth = num if num is not None else identity.get("top_k")
        return cls(str(identity["index"]), str(identity["encoder"]), depth,
                   normalise_query(query))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheKey":
        try:
            return cls(str(data["index"]), str(data["encoder"]), data.get("top_k"),
                       str(data["query"]))
        except (KeyError, TypeError) as error:
            raise ValueError(f"not a cache key: {data!r}") from error

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "encoder": self.encoder, "top_k": self.top_k,
                "query": self.query}


class CacheStore:
    """A JSONL store, one line per key; a torn or unreadable line is skipped."""

    def __init__(self, path: Optional[Any] = None,
                 warn: Callable[[str], None] = warnings.warn) -> None:
        self.path = Path(path) if path is not None else None
        self._warn = warn
        self._entries: Dict[CacheKey, Tuple[List[dict], Optional[List[float]]]] = {}
        self._load()

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, key: CacheKey) -> bool:
        return key in self._entries

    def get(self, key: CacheKey) -> Optional[Tuple[List[dict], Optional[List[float]]]]:
        """A deep copy of the entry, so callers can never mutate the cache."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        docs, scores = entry
        return copy.deepcopy(docs), copy.deepcopy(scores)

    def put(self, key: CacheKey, docs: List[dict],
            scores: Optional[List[float]]) -> None:
        stored = (copy.deepcopy(docs), copy.deepcopy(scores))
        self._entries[key] = stored
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            append_line(self.path, {**key.as_dict(), "docs": stored[0],
                                    "scores": stored[1]})

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = self._record(line, number)
            if record is not None:
                self._entries[record[0]] = record[1:]

    def _record(self, line: str, number: int):
        try:
            data = json.loads(line)
            key = CacheKey.from_dict(data)
            docs, scores = data["docs"], data.get("scores")
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            self._warn(f"retrieval cache {self.path}: skipping line {number}: {error}")
            return None
        return key, docs, scores


def _accepts(func: Any, name: str) -> bool:
    """True if ``func`` takes a keyword called ``name`` (or accepts ``**kwargs``)."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(p.kind is p.VAR_KEYWORD
                                     for p in parameters.values())


class CachingRetriever:
    """The proxy. Both public methods resolve through ``batch_search``."""

    def __init__(self, inner: Any, *, index: str, encoder: str,
                 top_k: Optional[int], path: Optional[Any] = None,
                 store: Optional[CacheStore] = None,
                 warn: Callable[[str], None] = warnings.warn) -> None:
        for name in ("search", "batch_search"):
            if not callable(getattr(inner, name, None)):
                raise TypeError(f"the wrapped retriever has no {name}()")
        self.inner = inner
        self.identity = {"index": index, "encoder": encoder, "top_k": top_k}
        self.store = store if store is not None else CacheStore(path, warn)
        self._scored = _accepts(inner.batch_search, "return_score")

    def search(self, query: str, num: Optional[int] = None,
               return_score: bool = False) -> Any:
        """One query: literally the batch path with a single-element request."""
        return self.batch_search([query], num=num, return_score=return_score)[0]

    def batch_search(self, queries: List[str], num: Optional[int] = None,
                     return_score: bool = False) -> List[Any]:
        if return_score and not self._scored:
            raise ValueError("the wrapped retriever cannot return scores")
        queries = list(queries)
        keys = [CacheKey.build(self.identity, query, num) for query in queries]
        pending = self._pending(keys)
        if pending:
            self._fill(pending, num)
        return [self._result(key, return_score) for key in keys]

    def _pending(self, keys: List[CacheKey]) -> List[CacheKey]:
        seen, pending = set(), []
        for key in keys:
            if key not in seen and not self.store.contains(key):
                pending.append(key)
            seen.add(key)
        return pending

    def _fill(self, pending: List[CacheKey], num: Optional[int]) -> None:
        kwargs: Dict[str, Any] = {}
        if num is not None and _accepts(self.inner.batch_search, "num"):
            kwargs["num"] = num
        if self._scored:
            kwargs["return_score"] = True
        docs_list, scores_list = self._call([key.query for key in pending], kwargs)
        for key, docs, scores in zip(pending, docs_list, scores_list):
            self.store.put(key, docs, scores)

    def _call(self, queries: List[str], kwargs: Dict[str, Any]):
        results = (self.inner.batch_search(queries, **kwargs) if kwargs
                   else self.inner.batch_search(queries))
        if not self._scored:
            results = list(results)
            if len(results) != len(queries):
                raise ValueError(f"the wrapped retriever returned {len(results)} "
                                 f"results for {len(queries)} queries")
            return results, [None] * len(results)
        docs_list, scores_list = results
        if len(docs_list) != len(queries) or len(scores_list) != len(queries):
            raise ValueError(f"the wrapped retriever returned {len(docs_list)} documents "
                             f"and {len(scores_list)} score lists for {len(queries)} queries")
        return docs_list, scores_list

    def _result(self, key: CacheKey, return_score: bool) -> Any:
        entry = self.store.get(key)
        if entry is None:  # only reachable if _fill skipped a key
            raise KeyError(f"no cached retrieval for {key.query!r}")
        docs, scores = entry
        return (docs, scores) if return_score else docs
