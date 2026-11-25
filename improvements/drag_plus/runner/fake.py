"""A deterministic offline backend: no model, no index, no network.

Outputs are a pure function of the prompt text, so the same question always produces the
same debate, which lets tests compare a resumed run with an uninterrupted one. The
answers are meaningless; this backend exists to exercise the path, never to give results.
``generate_with_scores`` (J24) returns hashed label log-probabilities, so the log-prob
Judge and the lazy Opponent run offline too.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

from ..adapters import prompt_count
from ..analysis.score import score_answer
from ..judge.logprob import ScoredOutput
from .label_scores import fake_label_logprobs


def _bucket(text: str, size: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % size


class FakeGenerator:
    """Pick a reply by the role markers the baseline prompts contain."""

    def generate(self, prompts: Any, **params: Any) -> List[str]:
        batch = prompts if prompt_count(prompts) > 1 else [prompts]
        return [self._reply(_joined(prompt)) for prompt in batch]

    def generate_with_scores(self, prompts: Any, labels: Sequence[str],
                             **params: Any) -> List[ScoredOutput]:
        """The highest-scoring label as text, with every label's hashed log-prob."""
        batch = prompts if prompt_count(prompts) > 1 else [prompts]
        outputs = []
        for prompt in batch:
            scores = fake_label_logprobs(_joined(prompt), labels)
            best = max(labels, key=lambda label: scores[label]) if labels else ""
            outputs.append(ScoredOutput(best, scores))
        return outputs

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


def _joined(prompt: Any) -> str:
    """A text prompt as is; a list of strings or chat messages joined by newlines."""
    if isinstance(prompt, str):
        return prompt
    return "\n".join(part["content"] if isinstance(part, dict) else part for part in prompt)


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
