"""Per-call timing rows (architecture §5, ``stage-timings.jsonl``).

A stage describes a call with a ``CallInfo``; ``Backends`` hands it to a ``CallRecorder``
only when profiling is on, so a run with ``profile_calls`` off makes the bare baseline
call and pays one ``if``. Rows land in the question's own ``calls`` list; the runner adds
run, variant and question columns when it writes the log.

``finish_reason`` is the server's own when the generator reports one (J24: the local
Ollama backend, and the vLLM log-prob scorer, through ``last_finish_reasons``). A batch
row is ``length`` if any prompt was cut, otherwise the shared reason, otherwise
``mixed``. FlashRAG's text-only ``generate`` reports none, so J08's inference applies
there: ``length`` when the output reaches ``max_tokens``, ``stop`` when a budget is set
but not reached, and null without a budget (the baseline call).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

Reasons = Optional[Callable[[], Optional[Sequence[str]]]]

STAGES = ("retrieval", "query_agent", "query_judge", "query_judge_retry",
          "answer_agent", "answer_moderator", "rerank", "gate")
ROW_FIELDS = ("stage", "round", "role", "batch_size", "prompt_tokens", "output_tokens",
              "finish_reason", "cache_hit", "t_start", "elapsed_ms")
LOG_FIELDS = ("run_id", "variant", "arm", "dataset", "split", "id") + ROW_FIELDS


@dataclass(frozen=True)
class CallInfo:
    stage: str
    round: int
    role: str
    sink: List[dict]


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _finish_reason(output_tokens: int, limit: Optional[int]) -> Optional[str]:
    """Infer the reason from a per-call budget; FlashRAG returns text only."""
    if limit is None:
        return None
    return "length" if output_tokens >= limit else "stop"


def observed_reason(reasons: Optional[Sequence[str]]) -> Optional[str]:
    """One row's reason from the server's per-prompt reasons; None when there are none."""
    if not reasons:
        return None
    if "length" in reasons:
        return "length"
    return reasons[0] if len(set(reasons)) == 1 else "mixed"


def _batch_finish_reason(tokens: List[int], limit: Optional[int]) -> Optional[str]:
    """One row per batch: report ``length`` if any output reached its cap."""
    reasons = [_finish_reason(count, limit) for count in tokens]
    if not reasons or reasons[0] is None:
        return None
    return "length" if "length" in reasons else "stop"


class CallRecorder:
    """Time a call with an injected clock and token counter, and append one row."""

    def __init__(self, clock: Callable[[], float],
                 count_tokens: Callable[[str], int] = whitespace_tokens):
        self._clock = clock
        self._count = count_tokens

    def generation(self, info: CallInfo, prompt: Any, call: Callable[[], str],
                   limit: Optional[int] = None, reasons: Reasons = None) -> str:
        start = self._clock()
        output = call()
        output_tokens = self._count(_as_text(output))
        reason = observed_reason(reasons() if reasons else None)
        row = self._row(info, start, 1, self._count(_as_text(prompt)), output_tokens,
                        reason or _finish_reason(output_tokens, limit))
        info.sink.append(row)
        return output

    def generation_batch(self, info: CallInfo, prompts: List[Any],
                         call: Callable[[], List[str]],
                         limit: Optional[int] = None, reasons: Reasons = None) -> List[str]:
        """One backend call for several prompts: one row, ``batch_size`` = prompt count."""
        start = self._clock()
        outputs = call()
        output_tokens = [self._count(_as_text(output)) for output in outputs]
        reason = observed_reason(reasons() if reasons else None)
        row = self._row(info, start, len(outputs),
                        sum(self._count(_as_text(prompt)) for prompt in prompts),
                        sum(output_tokens),
                        reason or _batch_finish_reason(output_tokens, limit))
        info.sink.append(row)
        return outputs

    def retrieval(self, info: CallInfo, batch_size: int, call: Callable[[], Any]) -> Any:
        start = self._clock()
        result = call()
        info.sink.append(self._row(info, start, batch_size, None, None))
        return result

    def _row(self, info: CallInfo, start: float, batch_size: int,
             prompt_tokens: Optional[int], output_tokens: Optional[int],
             finish_reason: Optional[str] = None) -> dict:
        elapsed_ms = round((self._clock() - start) * 1000.0, 3)
        return {"stage": info.stage, "round": info.round, "role": info.role,
                "batch_size": batch_size, "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens, "finish_reason": finish_reason,
                "cache_hit": False, "t_start": start, "elapsed_ms": elapsed_ms}
