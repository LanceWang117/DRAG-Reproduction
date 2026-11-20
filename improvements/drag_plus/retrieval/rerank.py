"""Optional cross-encoder reranking; ``sentence-transformers`` is imported on first use.

The interface is one injected scorer: it takes ``(query, passage)`` string pairs and
returns one score each, higher meaning more relevant. Tests inject a deterministic
scorer, so this module and the retriever that uses it never import torch or a model
unless a run actually sets ``rerank_k``.

``Drowning in Documents`` (Jacob et al., arXiv:2411.11767) shows reranker effectiveness
declining as more documents are scored, so the caller passes a bounded candidate list
and this class does not decide the depth.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

Scorer = Callable[[Sequence[Tuple[str, str]]], Sequence[float]]


def load_cross_encoder(model_name: str, batch_size: int = 32) -> Scorer:
    """Build a ``CrossEncoder`` scorer; the import stays inside this function."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    return lambda pairs: model.predict(list(pairs), batch_size=batch_size)


class CrossEncoderReranker:
    """Reorder passages by a cross-encoder score, deterministically on ties."""

    def __init__(self, model_name: str, *, scorer: Optional[Scorer] = None,
                 batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._scorer = scorer

    def _score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if self._scorer is None:
            self._scorer = load_cross_encoder(self.model_name, self.batch_size)
        scores = [float(score) for score in self._scorer(pairs)]
        if len(scores) != len(pairs):
            raise ValueError(f"the reranker scored {len(scores)} of {len(pairs)} pairs")
        return scores

    def rerank(self, query: str, docs: Sequence[dict]) -> List[Tuple[dict, float]]:
        """``(passage, score)`` pairs, best first, ties broken on the passage id."""
        if not docs:
            return []
        pairs = [(query, str(doc.get("contents", ""))) for doc in docs]
        scores = self._score(pairs)
        order = sorted(range(len(docs)),
                       key=lambda index: (-scores[index], str(docs[index].get("id"))))
        return [(docs[index], scores[index]) for index in order]
