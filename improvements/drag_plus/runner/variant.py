"""The variant file: a name, the flags to set and the arms to run.

Anything unrecognised is an error, like an unknown flag, so a typo cannot silently run
the baseline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from ..flags import Flags

ARMS = ("drag", "naive")
ALLOWED_ARMS = ARMS + ("ircot", "selfask")
_KEYS = {"name", "description", "flags", "arms"}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class VariantError(ValueError):
    """The variant file is malformed."""


@dataclass(frozen=True)
class Variant:
    name: str
    flags: Flags
    arms: Tuple[str, ...]
    raw: Dict[str, Any]


def _arms(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise VariantError("'arms' must be a non-empty list")
    bad = [arm for arm in value if arm not in ALLOWED_ARMS]
    if bad or len(set(value)) != len(value):
        raise VariantError(
            f"'arms' must be distinct names from {list(ALLOWED_ARMS)}, got {value!r}")
    return tuple(value)


def parse_variant(data: Any) -> Variant:
    """Validate one variant document, including its allowed arms."""
    if not isinstance(data, dict):
        raise VariantError("a variant file must hold a JSON object")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise VariantError(f"unknown variant key(s) {unknown}; known keys: {sorted(_KEYS)}")
    name = data.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise VariantError(f"'name' must match {_NAME.pattern}, got {name!r}")
    flags = data.get("flags", {})
    if not isinstance(flags, dict):
        raise VariantError("'flags' must be a JSON object")
    return Variant(name, Flags(flags), _arms(data.get("arms", list(ARMS))), dict(data))


def load_variant(path: Any) -> Variant:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise VariantError(f"variant file {str(path)!r} is not valid JSON: {error}") from error
    return parse_variant(data)
