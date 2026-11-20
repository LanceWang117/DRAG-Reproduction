"""``--backend local``: a real small model through Ollama, stored passages for retrieval.

For smoke runs only: it catches prompt, parsing and stage-flow bugs on real model text
before the A100 stage. Its outputs come from a different model family and size than the
A100's Llama-3.1-8B BF16, so they are never results and never comparable with them.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..retrieval.cache import index_identity
from .backends import Backend
from .fake import fake_evaluate
from .local_generator import (DEFAULT_MODEL, DEFAULT_URL, OllamaGenerator,
                              estimate_tokens)
from .local_retrieval import TOP_K, ReplayRetriever, load_replay

NOT_COMPARABLE = "smoke only: a small local model, not comparable with the A100 runs"
_ANSWER_RULE = "Always put the answer after 'The answer is: ', e.g.'The answer is: {example}.', "
_YES_NO_RULE = "Given four answer candidates, Yes and No, choose the best answer choice. "


class MessageTemplate:
    """Message lists for the injected generator, never a chat-formatted string.

    DRAG stages hand over finished messages, which pass through as copies. The Naive arm
    gets the frozen standard-RAG prompt for the dataset, with the reference formatted as
    FlashRAG's ``PromptTemplate.format_reference`` does (flashrag-dev 0.1.4.dev20250111).
    """

    def __init__(self, dataset: str):
        self._yes_no = dataset == "StrategyQA"

    def get_string(self, messages: Optional[List[dict]] = None, question: Any = None,
                   retrieval_result: Any = None, **params: Any) -> List[dict]:
        if messages is not None:
            return copy.deepcopy(messages)
        return [{"role": "system", "content": self._system(retrieval_result or [])},
                {"role": "user", "content": f"Question: {question}"}]

    def _system(self, retrieval_result: List[dict]) -> str:
        rule = _ANSWER_RULE.format(example="Yes" if self._yes_no else "answer")
        return ("Answer the question based on the given document. "
                + (_YES_NO_RULE if self._yes_no else "") + rule + "at the end of your response. "
                "\nThe following are given documents.\n" + _format_reference(retrieval_result))


def _format_reference(retrieval_result: List[dict]) -> str:
    lines = []
    for index, doc in enumerate(retrieval_result):
        title, _, text = doc["contents"].partition("\n")
        lines.append(f"Doc {index + 1}(Title: {title}) {text}\n")
    return "".join(lines)


def _describe(generator: OllamaGenerator, retriever: ReplayRetriever,
              replay: Path) -> Dict[str, Any]:
    return {"gpu": "none (Ollama server on this machine)", "model": f"ollama:{generator.model}",
            "index": f"replay:{replay.name}", "seed": generator.seed,
            "base_url": generator.base_url, "note": NOT_COMPARABLE,
            "generator": generator.stats.as_dict(), "retrieval": retriever.stats}


def build_local_backend(*, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
                        retrieval_path: Path, questions_path: Path,
                        generator: Optional[OllamaGenerator] = None) -> Backend:
    """Compose the local backend; ``generator`` lets a test inject its own."""
    generator = generator or OllamaGenerator(model, base_url)
    retriever = load_replay(Path(retrieval_path), Path(questions_path))
    replay = Path(retrieval_path)
    return Backend(
        "local", generator, retriever, lambda arm, dataset: MessageTemplate(dataset),
        fake_evaluate, lambda: _describe(generator, retriever, replay),
        lambda: {"index": index_identity(replay), "encoder": "replay", "top_k": TOP_K},
        estimate_tokens)
