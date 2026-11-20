"""Per-call timing rows (architecture §5, ``stage-timings.jsonl``).

A stage describes a call with a ``CallInfo``; ``Backends`` hands it to a ``CallRecorder``
only when profiling is on, so a run with ``profile_calls`` off makes the bare baseline
call and pays one ``if``. Rows land in the question's own ``calls`` list; the runner adds
run, variant and question columns when it writes the log.

Not captured (marked IC in the research note): vLLM request metrics and finish reasons,
because FlashRAG's generator returns text only. ``finish_reason`` is therefore null.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

STAGES = ("retrieval", "query_agent", "query_judge", "query_judge_retry",
          "answer_agent", "answer_moderator", "rerank", "gate")
ROW_FIELDS = ("stage", "round", "role", "batch_size", "prompt_tokens", "output_tokens",
              "finish_reason", "cache_hit", "t_start", "elapsed_ms")
LOG_FIELDS = ("run_id", "variant", "dataset", "split", "id") + ROW_FIELDS


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


class CallRecorder:
    """Time a call with an injected clock and token counter, and append one row."""

    def __init__(self, clock: Callable[[], float],
                 count_tokens: Callable[[str], int] = whitespace_tokens):
        self._clock = clock
        self._count = count_tokens

    def generation(self, info: CallInfo, prompt: Any, call: Callable[[], str]) -> str:
        start = self._clock()
        output = call()
        row = self._row(info, start, 1, self._count(_as_text(prompt)),
                        self._count(_as_text(output)))
        info.sink.append(row)
        return output

    def retrieval(self, info: CallInfo, batch_size: int, call: Callable[[], Any]) -> Any:
        start = self._clock()
        result = call()
        info.sink.append(self._row(info, start, batch_size, None, None))
        return result

    def _row(self, info: CallInfo, start: float, batch_size: int,
             prompt_tokens: Optional[int], output_tokens: Optional[int]) -> dict:
        elapsed_ms = round((self._clock() - start) * 1000.0, 3)
        return {"stage": info.stage, "round": info.round, "role": info.role,
                "batch_size": batch_size, "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens, "finish_reason": None,
                "cache_hit": False, "t_start": start, "elapsed_ms": elapsed_ms}
