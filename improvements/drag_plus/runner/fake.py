"""A deterministic offline backend: no model, no index, no network.

Outputs are a pure function of the prompt text, so the same question always produces the
same debate, which lets tests compare a resumed run with an uninterrupted one. The
answers are meaningless; this backend exists to exercise the path, never to give results.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from ..analysis.score import score_answer


def _bucket(text: str, size: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % size


class FakeGenerator:
    """Pick a reply by the role markers the baseline prompts contain."""

    def generate(self, prompts: Any, **params: Any) -> List[str]:
        text = prompts if isinstance(prompts, str) else "\n".join(prompts)
        return [self._reply(text)]

    @staticmethod
    def _reply(text: str) -> str:
        answer = f"answer {_bucket(text, 5)}"
        if "You are the judge in a debate" in text:
            return "Opponent" if _bucket(text, 3) == 0 else "Proponent"
        if "critical thinker and debater" in text:
            return f"The evidence is thin. Query Expansion: details {_bucket(text, 4)}"
        if "You are a debater" in text:
            return "The retrieved content is sufficient."
        if "moderator in a debate competition" in text:
            return f"The answer is: {answer}"
        return f"Reasoning from the documents. The answer is: {answer}"


class FakeRetriever:
    def search(self, query: str) -> List[dict]:
        return [{"id": f"fake-{_bucket(query, 1000)}-{rank}",
                 "contents": f"Title {rank}\nPassage {rank} about {query}"} for rank in range(2)]

    def batch_search(self, queries: List[str]) -> List[List[dict]]:
        return [self.search(query) for query in queries]


class FakeTemplate:
    """Chat messages and Naive prompts rendered as plain text."""

    def get_string(self, messages: Any = None, question: Any = None,
                   retrieval_result: Any = None, **params: Any) -> str:
        if messages is not None:
            return "\n".join(m["content"] for m in messages)
        docs = "\n".join(d["contents"] for d in retrieval_result or [])
        return f"Answer from the documents.\n{docs}\nQuestion: {question}"


def fake_evaluate(arm: str, dataset: str, records: List[Dict[str, Any]]) -> Dict[str, float]:
    scores = [score_answer(r["pred"], r["golden_answers"]) for r in records]
    return {metric: sum(s[metric] for s in scores) / len(scores) for metric in scores[0]}


def fake_description() -> Dict[str, str]:
    return {"gpu": "none (fake backend)", "model": "fake-generator", "index": "fake-retriever", "seed": None}
