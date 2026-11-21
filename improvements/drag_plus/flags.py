"""Validated flag registry. Every default reproduces the v1.0.1 baseline.

A flag is one row in ``FLAG_SPECS``; a later job adds its flags by appending rows. An
unknown name, a value of the wrong type or a value outside the allowed set is a
``FlagError``, so a typo in a variant file can never silently run the baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


class FlagError(ValueError):
    """A variant file or caller supplied an unknown or invalid flag."""


@dataclass(frozen=True)
class FlagSpec:
    name: str
    kind: type
    default: Any
    choices: Optional[Tuple[Any, ...]] = None
    minimum: Optional[int] = None
    doc: str = ""
    optional: bool = False


FLAG_SPECS: Tuple[FlagSpec, ...] = (
    FlagSpec("max_query_debate_rounds", int, 3, minimum=0,
             doc="Query-stage debate rounds; 0 retrieves for the question only."),
    FlagSpec("max_answer_debate_rounds", int, 3, minimum=0,
             doc="Answer-stage debate rounds; 0 answers directly from the pool."),
    FlagSpec("profile_calls", bool, False,
             doc="Record one timing row per generator and retriever call."),
    FlagSpec("answer_extractor", str, "native", choices=("native", "unified"),
             doc="native keeps the raw moderator text as pred; unified applies E(.)."),
    FlagSpec("moderator_final_only", bool, False,
             doc="Answer moderator runs once, after the last round (paper Eq. 13)."),
    FlagSpec("batch_agents_in_round", bool, False,
             doc="Send both agents' prompts of each round in one 2-prompt call."),
    FlagSpec("retrieval_cache", str, None, optional=True,
             doc="Path to retrieval-cache.jsonl; null disables the caching proxy."),
    FlagSpec("retriever", str, "e5", choices=("e5", "hybrid"),
             doc="e5 keeps the dense retriever; hybrid fuses it with BM25 by weighted RRF."),
    FlagSpec("dense_top_k", int, 20, minimum=1,
             doc="Dense candidates per query before fusion (hybrid only)."),
    FlagSpec("bm25_top_k", int, 20, minimum=1,
             doc="BM25 candidates per query before fusion (hybrid only)."),
    FlagSpec("hybrid_top_k", int, 3, minimum=1,
             doc="Passages the hybrid retriever returns; 3 matches the recorded baseline."),
    FlagSpec("rrf_k", int, 60, minimum=1,
             doc="RRF rank constant; Cormack et al. (2009) fixed k=60."),
    FlagSpec("dense_weight", float, 1.0, minimum=0,
             doc="RRF weight of the dense ranking; 0 drops it."),
    FlagSpec("bm25_weight", float, 1.0, minimum=0,
             doc="RRF weight of the BM25 ranking; 0 drops it."),
    FlagSpec("dense_score_floor", float, None, optional=True,
             doc="Drop dense candidates scoring below this before fusion."),
    FlagSpec("bm25_score_floor", float, None, optional=True,
             doc="Drop BM25 candidates scoring below this before fusion."),
    FlagSpec("bm25_index", str, None, optional=True,
             doc="Path to a bm25s index directory saved with its corpus; hybrid only."),
    FlagSpec("rerank_k", int, None, optional=True, minimum=1,
             doc="Rerank the first N fused candidates with a cross-encoder; null disables it."),
    FlagSpec("rerank_model", str, "BAAI/bge-reranker-base",
             doc="Cross-encoder id used when rerank_k is set."),
    FlagSpec("per_title_cap", int, None, optional=True, minimum=1,
             doc="Keep at most this many passages per article title; null keeps all."),
    FlagSpec("skip_duplicate_query", bool, False,
             doc="Skip the search and end the query stage when a proposed query is "
                 "effectively already in the pool (normalised token set)."),
    FlagSpec("zero_novelty_exit", bool, False,
             doc="End the query stage when a search returns no passage ID the pool "
                 "did not already have before the action."),
)
_SPECS_BY_NAME = {spec.name: spec for spec in FLAG_SPECS}


def _check(spec: FlagSpec, value: Any) -> Any:
    # bool is an int subclass, so True would otherwise pass for an integer flag.
    if value is None and spec.optional:
        return None
    if spec.kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)  # 1 and 1.0 are the same weight; "1" is still an error
    if isinstance(value, bool) != (spec.kind is bool) or not isinstance(value, spec.kind):
        raise FlagError(
            f"flag {spec.name!r} expects {spec.kind.__name__}, "
            f"got {type(value).__name__}: {value!r}")
    if spec.choices is not None and value not in spec.choices:
        raise FlagError(f"flag {spec.name!r} must be one of {list(spec.choices)}, got {value!r}")
    if spec.kind is str and not value.strip():
        raise FlagError(f"flag {spec.name!r} must not be blank")
    if spec.minimum is not None and value < spec.minimum:
        raise FlagError(f"flag {spec.name!r} must be >= {spec.minimum}, got {value!r}")
    return value


class Flags:
    """Immutable, validated flag values; unspecified flags take the baseline default."""

    def __init__(self, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        unknown = sorted(set(values) - set(_SPECS_BY_NAME))
        if unknown:
            raise FlagError(
                f"unknown flag(s) {unknown}; known flags: {sorted(_SPECS_BY_NAME)}")
        resolved = {s.name: _check(s, values.get(s.name, s.default)) for s in FLAG_SPECS}
        object.__setattr__(self, "_values", resolved)

    def __getattr__(self, name: str) -> Any:
        values = self.__dict__.get("_values", {})
        if name in values:
            return values[name]
        raise AttributeError(f"unknown flag {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Flags are immutable; build a new Flags with with_changes()")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Flags) and self._values == other._values

    def __hash__(self) -> int:
        return hash(tuple(self._values.items()))

    def __repr__(self) -> str:
        return f"Flags({self._values!r})"

    def with_changes(self, **changes: Any) -> "Flags":
        return Flags({**self._values, **changes})

    def to_manifest(self) -> dict:
        """Every flag with its resolved value, for ``run-manifest.json``."""
        return dict(self._values)


def load_flags(path: Any) -> Flags:
    """Read flags from a JSON object file; a non-object document is an error."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise FlagError(f"flag file {str(path)!r} must hold a JSON object")
    return Flags(data)
