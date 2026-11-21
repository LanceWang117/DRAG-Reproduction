"""What a rehearsal runs: dev questions, which arms can run locally, and guards.

* Questions: the first k per dataset in manifest order, so a rehearsal is a prefix of
  E00's 10-per-dataset subset. k = N / number of datasets.
* Arms: an arm whose flags need A100-only machinery is skipped with its reasons, never
  run without its feature. Server tuning flags run with a "no local effect" note.
* Guards: only the dev manifest is accepted (the test set is touched once, on the A100,
  in E06), and two variants may not share a name (their run directories would collide).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

MIN_QUESTIONS, MAX_QUESTIONS = 6, 30

Rule = Tuple[str, Callable[[Any], bool], str]

_NOT_NONE = lambda value: value not in (None, "none")  # noqa: E731
SKIP_RULES: Tuple[Rule, ...] = (
    ("retriever", lambda value: value == "hybrid",
     "hybrid retrieval needs the server's dense and BM25 indexes"),
    ("bm25_index", lambda value: value is not None, "needs the server's BM25 index"),
    ("rerank_k", lambda value: value is not None, "needs the cross-encoder reranker (torch)"),
    ("faiss_gpu", bool, "GPU FAISS runs only on the A100"),
    ("enable_prefix_caching", bool, "vLLM prefix caching runs only on the A100"),
    ("quantization", _NOT_NONE, "quantised vLLM weights run only on the A100"),
    ("speculative", _NOT_NONE, "vLLM speculative decoding runs only on the A100"),
)
NOTE_RULES: Tuple[Rule, ...] = (
    ("gpu_memory_utilization", lambda value: value is not None, "no local effect (server tuning)"),
    ("omp_threads", lambda value: value is not None, "no local effect (server tuning)"),
    ("question_batch_size", lambda value: isinstance(value, int) and value > 1,
     "Ollama batches by its own rules: structure is valid, throughput is not"),
)


class RehearsalError(ValueError):
    """The rehearsal request is invalid or would touch data it must not use."""


@dataclass(frozen=True)
class ArmPlan:
    name: str
    runnable: bool
    reasons: Tuple[str, ...]
    notes: Tuple[str, ...]


def check_manifest(path: Path, dev_manifest: Path) -> None:
    """Accept only the dev manifest; refuse the test set with an explicit reason."""
    resolved = Path(path).resolve()
    if any(part.startswith("J05") for part in resolved.parts):
        raise RehearsalError(f"refusing {path}: the test set is used once, on the A100, in E06")
    if resolved != Path(dev_manifest).resolve():
        raise RehearsalError(f"refusing {path}: rehearsals use the dev manifest {dev_manifest}")


def _datasets_in_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys(row["dataset"] for row in rows))


def select_rows(rows: Sequence[Dict[str, Any]], questions: int) -> List[Dict[str, Any]]:
    """The first questions/len(datasets) rows of every dataset, in manifest order."""
    datasets = _datasets_in_order(rows)
    if not MIN_QUESTIONS <= questions <= MAX_QUESTIONS or questions % len(datasets):
        raise RehearsalError(f"--questions must be a multiple of {len(datasets)} "
                             f"between {MIN_QUESTIONS} and {MAX_QUESTIONS}, got {questions}")
    per_dataset = questions // len(datasets)
    available = Counter(row["dataset"] for row in rows)
    short = sorted(name for name in datasets if available[name] < per_dataset)
    if short:
        raise RehearsalError(f"fewer than {per_dataset} dev questions in {short}")
    taken: Counter = Counter()
    selected = []
    for row in rows:
        if taken[row["dataset"]] < per_dataset:
            taken[row["dataset"]] += 1
            selected.append(row)
    return selected


def _matches(rules: Sequence[Rule], flags: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(f"{name}={flags[name]!r}: {text}" for name, applies, text in rules
                 if name in flags and applies(flags[name]))


def classify(name: str, raw: Dict[str, Any]) -> ArmPlan:
    """Decide whether one variant can run locally, from the flags its file sets."""
    flags = raw.get("flags", {})
    reasons = _matches(SKIP_RULES, flags)
    return ArmPlan(name, not reasons, reasons, _matches(NOTE_RULES, flags))


def plan_arms(variants: Sequence[Any]) -> List[ArmPlan]:
    """One plan per variant, in order; duplicate names are refused."""
    duplicates = sorted(n for n, count in Counter(v.name for v in variants).items() if count > 1)
    if duplicates:
        raise RehearsalError(f"variant names must be unique, repeated: {duplicates}")
    return [classify(variant.name, variant.raw) for variant in variants]
