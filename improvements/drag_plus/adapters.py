"""Protocols for the objects the runner injects, and a thin call wrapper.

``typing.Protocol`` is structural: FlashRAG classes and test fakes satisfy it without
inheriting from anything, so this package never imports FlashRAG, vLLM, torch or FAISS.
``runtime_checkable`` plus a callable check lets the pipeline reject a missing method at
construction rather than part-way through a run on the server (names only, not signatures).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

from .flags import Flags
from .instrument.timing import CallInfo, CallRecorder, whitespace_tokens
from .judge.logprob import ScoredOutput


class GeneratorContractError(ValueError):
    """A generator returned a different number of outputs than prompts."""


BATCH_ROLE = "agents"
"""Timing-row role of a batched two-agent call: one backend call, two prompts."""


def last_finish_reasons(generator: Any, count: int) -> Optional[List[str]]:
    """The generator's per-prompt reasons for its latest call, if it reports ``count``."""
    reasons = getattr(generator, "last_finish_reasons", None)
    if not isinstance(reasons, list) or len(reasons) != count:
        return None
    return list(reasons) if all(isinstance(r, str) for r in reasons) else None


def prompt_count(prompts: Any) -> int:
    """How many prompts one ``generate`` call carries.

    A single prompt is a string or a message list; a batch is a list of prompts, so
    its first element is not a message dict.
    """
    if isinstance(prompts, list) and prompts and not isinstance(prompts[0], dict):
        return len(prompts)
    return 1


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
    recorder: Optional[CallRecorder] = None

    def __post_init__(self) -> None:
        for name, protocol, method in (("generator", Generator, "generate"),
                                       ("retriever", Retriever, "search"),
                                       ("prompt_template", PromptTemplate, "get_string")):
            backend = getattr(self, name)
            if not isinstance(backend, protocol) or not callable(getattr(backend, method)):
                raise TypeError(f"{name} does not implement {protocol.__name__}")

    def render(self, messages: list) -> Any:
        return self.prompt_template.get_string(messages=messages)

    def complete(self, prompt: Any, *, call: Optional[CallInfo] = None, **params: Any) -> str:
        """One generation. Params are forwarded only when given (baseline passes none)."""
        def invoke() -> str:
            return self.generator.generate(prompt, **params)[0]
        if self.recorder is None or call is None:
            return invoke()
        return self.recorder.generation(call, prompt, invoke, limit=params.get("max_tokens"),
                                        reasons=self._reasons(1))

    def complete_batch(self, prompts: List[Any], *, call: Optional[CallInfo] = None,
                       **params: Any) -> List[str]:
        """One generation returning ``len(prompts)`` outputs, mapped back by position."""
        def invoke() -> List[str]:
            outputs = list(self.generator.generate(prompts, **params))
            if len(outputs) != len(prompts):
                raise GeneratorContractError(
                    f"generator returned {len(outputs)} outputs for {len(prompts)} prompts")
            return outputs
        if self.recorder is None or call is None:
            return invoke()
        return self.recorder.generation_batch(call, prompts, invoke,
                                              limit=params.get("max_tokens"),
                                              reasons=self._reasons(len(prompts)))

    def complete_with_scores(self, prompts: List[Any], *, labels,
                             call: Optional[CallInfo] = None,
                             **params: Any) -> List[ScoredOutput]:
        """One scored generation (J12): text plus a log-prob per label and prompt.

        The injected generator must implement ``generate_with_scores``; the baseline
        ``generate`` contract is unchanged. Profiling records the call once, with the
        generated texts for token counts, as for a text batch.
        """
        score = getattr(self.generator, "generate_with_scores", None)
        if not callable(score):
            raise TypeError(
                "judge_scoring='logprob' needs a generator with "
                "generate_with_scores(prompts, labels=...)")

        def invoke() -> List[ScoredOutput]:
            outputs = list(score(prompts, labels=labels, **params))
            if len(outputs) != prompt_count(prompts):
                raise GeneratorContractError(
                    f"generator returned {len(outputs)} scored outputs for "
                    f"{prompt_count(prompts)} prompts")
            return outputs

        if self.recorder is None or call is None:
            return invoke()
        captured: List[ScoredOutput] = []

        def timed() -> List[str]:
            captured.extend(invoke())
            return [output.text for output in captured]

        batch = prompts if isinstance(prompts, list) else [prompts]
        self.recorder.generation_batch(call, batch, timed, limit=params.get("max_tokens"),
                                       reasons=self._reasons(len(batch)))
        return captured

    def _reasons(self, count: int) -> Callable[[], Optional[List[str]]]:
        return lambda: last_finish_reasons(self.generator, count)

    def search(self, query: str, *, call: Optional[CallInfo] = None) -> List[dict]:
        if self.recorder is None or call is None:
            return self.retriever.search(query)
        return self.recorder.retrieval(call, 1, lambda: self.retriever.search(query))

    def batch_search(self, queries: List[str], *,
                     call: Optional[CallInfo] = None) -> List[List[dict]]:
        if self.recorder is None or call is None:
            return self.retriever.batch_search(queries)
        return self.recorder.retrieval(call, len(queries),
                                       lambda: self.retriever.batch_search(queries))


@dataclass(frozen=True)
class StageContext:
    """Run-wide, read-only inputs every stage receives next to the question state."""

    flags: Flags
    backends: Backends
    dataset_name: str
    count_tokens: Callable[[str], int] = whitespace_tokens
    gate_model: Optional[Any] = None

    @property
    def strategy_qa(self) -> bool:
        return self.dataset_name == "StrategyQA"
