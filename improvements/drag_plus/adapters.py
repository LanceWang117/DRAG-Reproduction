"""Protocols for the objects the runner injects, and a thin call wrapper.

``typing.Protocol`` is structural: FlashRAG classes and test fakes satisfy it without
inheriting from anything, so this package never imports FlashRAG, vLLM, torch or FAISS.
``runtime_checkable`` lets the pipeline reject a missing method at construction rather
than part-way through a run on the server (it checks method names, not signatures).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    def generate(self, prompts: Any, **params: Any) -> List[str]: ...


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str) -> List[dict]: ...


@runtime_checkable
class PromptTemplate(Protocol):
    def get_string(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Backends:
    """The three injected objects, called exactly as the frozen baseline calls them."""

    generator: Generator
    retriever: Retriever
    prompt_template: PromptTemplate

    def __post_init__(self) -> None:
        for name, protocol in (("generator", Generator), ("retriever", Retriever),
                               ("prompt_template", PromptTemplate)):
            if not isinstance(getattr(self, name), protocol):
                raise TypeError(f"{name} does not implement {protocol.__name__}")

    def render(self, messages: list) -> Any:
        return self.prompt_template.get_string(messages=messages)

    def complete(self, prompt: Any, **params: Any) -> str:
        """One generation. Params are forwarded only when given (baseline passes none)."""
        return self.generator.generate(prompt, **params)[0]

    def search(self, query: str) -> List[dict]:
        return self.retriever.search(query)
