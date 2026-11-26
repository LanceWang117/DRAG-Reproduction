"""The server-check registry (J26): load, validate, list and run.

``configs/server-checks.json`` lists every item the PC jobs could not verify, with a
tier (``auto`` probes, ``smoke`` one-question runs, ``measure`` inside an E-job), the
E-job that owns it, and the expected result. ``run_auto`` executes each distinct probe
once and maps the outcome back to every check that names it. A probe that raises is an
``error``, never a pass; server-only probes are ``skipped`` off the server.
"""
from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .probes import PROBES, ProbeContext

TIERS = ("auto", "smoke", "measure")
_KEYS = {"id", "job", "tier", "owner", "expect", "probe"}
_ID = re.compile(r"(J\d{2}[a-z]?)-\d+")
_OWNER = re.compile(r"E0\d[ab]?")


class RegistryError(ValueError):
    """The registry file is malformed."""


@dataclass(frozen=True)
class Check:
    id: str
    job: str
    tier: str
    owner: str
    expect: str
    probe: Optional[str] = None


def _check(entry: Any, index: int) -> Check:
    if not isinstance(entry, dict) or set(entry) - _KEYS:
        raise RegistryError(f"check #{index}: must be an object with keys from {sorted(_KEYS)}")
    check = Check(**{key: entry.get(key) for key in _KEYS})
    match = _ID.fullmatch(str(check.id))
    if not match or match.group(1) != check.job:
        raise RegistryError(f"check #{index}: id {check.id!r} must be '<job>-<n>' for {check.job!r}")
    if check.tier not in TIERS or not _OWNER.fullmatch(str(check.owner)):
        raise RegistryError(f"{check.id}: tier must be one of {TIERS} and owner an E-job")
    if not isinstance(check.expect, str) or not check.expect.strip():
        raise RegistryError(f"{check.id}: 'expect' must say what passing looks like")
    if (check.tier == "auto") != (check.probe is not None) or \
            (check.probe is not None and check.probe not in PROBES):
        raise RegistryError(f"{check.id}: auto checks name a known probe; others name none")
    return check


def load_checks(path: Path) -> List[Check]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistryError(f"{path} is not valid JSON: {error}") from error
    checks = [_check(entry, index) for index, entry in enumerate(data.get("checks", []))]
    ids = [check.id for check in checks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise RegistryError(f"duplicate check ids {duplicates}")
    return checks


def select(checks: Sequence[Check], tier: Optional[str] = None,
           owner: Optional[str] = None) -> List[Check]:
    return [c for c in checks if (tier is None or c.tier == tier)
            and (owner is None or c.owner == owner)]


def format_list(checks: Sequence[Check]) -> str:
    """A Markdown checklist grouped by owner, then tier."""
    lines = []
    for owner in sorted({c.owner for c in checks}):
        lines.append(f"## {owner}\n")
        for tier in TIERS:
            for check in [c for c in checks if c.owner == owner and c.tier == tier]:
                probe = f" (probe `{check.probe}`)" if check.probe else ""
                lines.append(f"- [ ] **{check.id}** [{tier}]{probe}: {check.expect}")
        lines.append("")
    return "\n".join(lines)


def _run_probe(name: str, ctx: ProbeContext) -> Dict[str, Any]:
    function, requires = PROBES[name]
    if requires == "server" and ctx.backend != "flashrag":
        return {"status": "skipped", "detail": f"needs the server stack, backend is {ctx.backend}",
                "seconds": 0.0}
    start = ctx.clock()
    try:
        ok, detail = function(ctx)
        status = "pass" if ok else "fail"
    except Exception as error:  # fail closed: an exception is an error, not a pass
        status = "error"
        detail = f"{type(error).__name__}: {error} | {traceback.format_exc(limit=2)[-300:]}"
    return {"status": status, "detail": detail, "seconds": round(ctx.clock() - start, 3)}


def run_auto(checks: Sequence[Check], ctx: ProbeContext) -> List[Dict[str, Any]]:
    """One result per auto check; each distinct probe runs once."""
    outcomes: Dict[str, Dict[str, Any]] = {}
    results = []
    for check in select(checks, tier="auto"):
        if check.probe not in outcomes:
            outcomes[check.probe] = _run_probe(check.probe, ctx)
        results.append({"id": check.id, "probe": check.probe, "owner": check.owner,
                        "expect": check.expect, **outcomes[check.probe]})
    return results


def summarise(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {status: 0 for status in ("pass", "fail", "error", "skipped")}
    for result in results:
        counts[result["status"]] += 1
    return counts


def passed(results: Sequence[Mapping[str, Any]]) -> bool:
    """True only when nothing failed or errored (skips are allowed off the server)."""
    counts = summarise(results)
    return counts["fail"] == 0 and counts["error"] == 0
