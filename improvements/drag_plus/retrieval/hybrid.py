"""Weighted reciprocal rank fusion of two retrievers, with floors and a title cap.

Cormack, Clarke and Büttcher (SIGIR 2009) score a document as
``sum_r 1 / (k + rank_r(d))`` with ``k = 60``; the weighted form used here multiplies
each ranking's contribution by ``w_r``. Ranks are 1-based and the fusion reads ranks, not
scores, so the two retrievers never have to be calibrated against each other.

Two knobs act before fusion and one after:

* ``floor`` drops a source's candidates whose own score is below it, so a retriever that
  is confidently wrong cannot pollute the fused list;
* ``cap`` keeps at most N passages per article, where the title is the first line of
  ``contents`` exactly as ``_format_reference`` reads it in the frozen baseline.

Ties break on the document id, so the same inputs always give the same order.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Ranked = List[Tuple[dict, float]]


@dataclass(frozen=True)
class Source:
    """One retriever's ranking, its optional per-passage scores and its fusion weight."""

    name: str
    docs: Sequence[dict]
    scores: Optional[Sequence[Optional[float]]] = None
    weight: float = 1.0
    floor: Optional[float] = None


def doc_id(doc: dict) -> str:
    """The identity a passage is fused and tie-broken on."""
    value = doc.get("id")
    if value is None:
        raise ValueError(f"a passage has no 'id': {doc!r}")
    return str(value)


def passage_title(contents: Any) -> str:
    """The baseline's title convention: the first line of ``contents``."""
    return str(contents).split("\n", 1)[0]


def _passes_floor(source: Source) -> Iterable[dict]:
    """The source's documents that clear its floor, each id once, in rank order."""
    if source.floor is None:
        seen = set()
        for doc in source.docs:
            if doc_id(doc) not in seen:
                seen.add(doc_id(doc))
                yield doc
        return
    if source.scores is None:
        raise ValueError(f"source {source.name!r} sets a score floor but has no scores")
    if len(source.scores) != len(source.docs):
        raise ValueError(f"source {source.name!r} has {len(source.docs)} documents and "
                         f"{len(source.scores)} scores")
    seen = set()
    for doc, score in zip(source.docs, source.scores):
        if score is not None and score >= source.floor and doc_id(doc) not in seen:
            seen.add(doc_id(doc))
            yield doc


def reciprocal_rank_fusion(sources: Sequence[Source], k: int = 60) -> Ranked:
    """Weighted RRF over any number of rankings; returns ``(document, score)`` pairs."""
    if k < 1:
        raise ValueError(f"RRF k must be >= 1, got {k}")
    total: Dict[str, float] = {}
    docs: Dict[str, dict] = {}
    for source in sources:
        if source.weight <= 0:
            continue
        for rank, doc in enumerate(_passes_floor(source), start=1):
            key = doc_id(doc)
            total[key] = total.get(key, 0.0) + source.weight / (k + rank)
            docs.setdefault(key, doc)
    order = sorted(total, key=lambda key: (-total[key], key))
    return [(docs[key], total[key]) for key in order]


def cap_per_title(ranked: Ranked, cap: Optional[int]) -> Ranked:
    """Keep at most ``cap`` passages per article title, preserving the input order."""
    if cap is None:
        return list(ranked)
    if cap < 1:
        raise ValueError(f"per_title_cap must be >= 1, got {cap}")
    counts: Dict[str, int] = {}
    kept: Ranked = []
    for doc, score in ranked:
        title = passage_title(doc.get("contents", ""))
        if counts.get(title, 0) < cap:
            counts[title] = counts.get(title, 0) + 1
            kept.append((doc, score))
    return kept


def accepts(func: Any, name: str) -> bool:
    """True if ``func`` takes a keyword called ``name``; ``**kwargs`` counts too."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(p.kind is p.VAR_KEYWORD
                                     for p in parameters.values())


def source_results(retriever: Any, queries: List[str], depth: int,
                   want_scores: bool) -> List[Tuple[List[dict], Optional[List[float]]]]:
    """Call a retriever for every query, asking for scores only when they are needed."""
    kwargs: Dict[str, Any] = {}
    if accepts(retriever.batch_search, "num"):
        kwargs["num"] = depth
    if want_scores:
        if not accepts(retriever.batch_search, "return_score"):
            raise ValueError(f"{type(retriever).__name__} cannot return scores, "
                             "which a score floor requires")
        kwargs["return_score"] = True
    results = retriever.batch_search(list(queries), **kwargs)
    if want_scores:
        if not isinstance(results, tuple) or len(results) != 2:
            raise TypeError("a retriever that accepts return_score must return "
                            "(documents, scores)")
        docs_list, scores_list = results
    else:
        docs_list, scores_list = results, [None] * len(queries)
    if len(docs_list) != len(queries) or len(scores_list) != len(queries):
        raise ValueError(f"{type(retriever).__name__} returned {len(docs_list)} document "
                         f"lists for {len(queries)} queries")
    return [(list(docs)[:depth], None if scores is None else list(scores)[:depth])
            for docs, scores in zip(docs_list, scores_list)]


def retrieval_config(flags: Any) -> Dict[str, Any]:
    """The retrieval settings a cache entry depends on besides index and query."""
    if flags.retriever == "e5":
        return {"retriever": "e5"}
    return {"retriever": "hybrid", "dense_top_k": flags.dense_top_k,
            "bm25_top_k": flags.bm25_top_k, "hybrid_top_k": flags.hybrid_top_k,
            "rrf_k": flags.rrf_k, "dense_weight": flags.dense_weight,
            "bm25_weight": flags.bm25_weight, "dense_score_floor": flags.dense_score_floor,
            "bm25_score_floor": flags.bm25_score_floor, "bm25_index": flags.bm25_index,
            "rerank_k": flags.rerank_k, "rerank_model": flags.rerank_model,
            "per_title_cap": flags.per_title_cap}


def config_fingerprint(flags: Any) -> str:
    """A canonical one-line form of :func:`retrieval_config` for the cache key."""
    return json.dumps(retrieval_config(flags), sort_keys=True, separators=(",", ":"))
