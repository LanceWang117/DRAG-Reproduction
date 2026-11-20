"""Replay retriever for the local smoke backend: stored passages, no index, no FAISS.

A query seen in the recorded 200-question run returns its stored passages. A new query
(a debate round rewrites queries the stored run never made) ranks every stored passage by
token overlap, so the smoke run still gets plausible text. Both paths are counted, so the
summary can say how much of a run was real replay.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..retrieval.cache import normalise_query

TOP_K = 3
_WORD = re.compile(r"\w+")
_STOP_WORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it of on or "
    "that the this to was were what when where which who whom why will with".split())


def _tokens(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower())) - _STOP_WORDS


class ReplayRetriever:
    """``search`` and ``batch_search`` over a map of stored query to passages."""

    def __init__(self, stored: Dict[str, List[dict]], top_k: int = TOP_K):
        self._top_k = top_k
        self._stored: Dict[str, List[dict]] = {}
        for query, passages in stored.items():
            self._stored.setdefault(normalise_query(query), passages)
        pool: Dict[Any, dict] = {}
        for passages in stored.values():
            for passage in passages:
                pool.setdefault(passage["id"], passage)
        self._pool = list(pool.values())
        self._pool_tokens = [_tokens(passage["contents"]) for passage in self._pool]
        self._replayed = self._lexical = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {"replayed": self._replayed, "lexical": self._lexical}

    def search(self, query: str) -> List[dict]:
        stored = self._stored.get(normalise_query(query))
        if stored is not None:
            self._replayed += 1
            return copy.deepcopy(stored[:self._top_k])
        self._lexical += 1
        return copy.deepcopy(self._rank(query))

    def batch_search(self, queries: List[str]) -> List[List[dict]]:
        return [self.search(query) for query in queries]

    def _rank(self, query: str) -> List[dict]:
        wanted = _tokens(query)
        scores = [len(wanted & tokens) for tokens in self._pool_tokens]
        order = sorted(range(len(self._pool)), key=lambda index: (-scores[index], index))
        return [self._pool[index] for index in order[:self._top_k]]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _question_texts(questions_path: Path) -> Dict[tuple, str]:
    return {(row["dataset"], row["split"], row["id"]): row["question"]
            for row in _read_jsonl(questions_path)}


def load_replay(retrieval_path: Path, questions_path: Path,
                top_k: Optional[int] = None) -> ReplayRetriever:
    """Build the retriever from ``results/retrieval-200.jsonl``.

    The stored DRAG map holds the queries the debate made; in some rows it lacks the
    question itself, so the Naive passages are keyed by the question text from the
    question file. A query stored on the DRAG side wins when both exist.
    """
    texts = _question_texts(Path(questions_path))
    stored: Dict[str, List[dict]] = {}
    for row in _read_jsonl(Path(retrieval_path)):
        for query, passages in row["drag"].items():
            stored.setdefault(query, passages)
        question = texts.get((row["dataset"], row["split"], row["id"]))
        if question is not None:
            stored.setdefault(question, row["naive"])
    return ReplayRetriever(stored, top_k or TOP_K)
