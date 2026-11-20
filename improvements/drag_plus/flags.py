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


FLAG_SPECS: Tuple[FlagSpec, ...] = (
    FlagSpec("max_query_debate_rounds", int, 3, minimum=0,
             doc="Query-stage debate rounds; 0 retrieves for the question only."),
    FlagSpec("max_answer_debate_rounds", int, 3, minimum=0,
             doc="Answer-stage debate rounds; 0 answers directly from the pool."),
    FlagSpec("profile_calls", bool, False,
             doc="Record one timing row per generator and retriever call."),
    FlagSpec("answer_extractor", str, "native", choices=("native", "unified"),
             doc="native keeps the raw moderator text as pred; unified applies E(.)."),
)
_SPECS_BY_NAME = {spec.name: spec for spec in FLAG_SPECS}


def _check(spec: FlagSpec, value: Any) -> Any:
    # bool is an int subclass, so True would otherwise pass for an integer flag.
    if isinstance(value, bool) != (spec.kind is bool) or not isinstance(value, spec.kind):
        raise FlagError(
            f"flag {spec.name!r} expects {spec.kind.__name__}, "
            f"got {type(value).__name__}: {value!r}")
    if spec.choices is not None and value not in spec.choices:
        raise FlagError(f"flag {spec.name!r} must be one of {list(spec.choices)}, got {value!r}")
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
