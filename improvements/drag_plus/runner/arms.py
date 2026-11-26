"""E-job arm files from one spec (J25).

``configs/arms/spec.json`` names flag layers and, per job and arm, the layers to merge in
order (a later layer wins, key by key). ``configs/arms/measured.json`` holds the layers
only the server can decide: E01's promoted configuration ``P``, E00's best thread count
and E04's best hybrid settings, each with ``provisional`` and ``source``. Every generated
file is validated as a variant, and its description names the job, arm, layers and any
provisional layer, so a run manifest shows exactly what ran.

Generated files live at ``configs/arms/<job>/<arm>.json``. ``check`` reports files that
are missing, differ from the spec, or are no longer produced by it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .variant import ARMS, VariantError, parse_variant

ROOT_FILES = ("spec.json", "measured.json", "README.md")


class ArmSpecError(ValueError):
    """The arm spec or the measured layers are malformed."""


@dataclass(frozen=True)
class Layer:
    flags: Mapping[str, Any]
    provisional: bool = False
    source: str = ""


def _object(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ArmSpecError(f"{where} must be a JSON object")
    return value


def layers_from(spec: Mapping[str, Any], measured: Mapping[str, Any]) -> Dict[str, Layer]:
    """Plain layers from the spec plus measured layers; a name may not appear twice."""
    plain = _object(spec.get("layers", {}), "spec 'layers'")
    layers = {name: Layer(dict(_object(flags, f"layer {name!r}")))
              for name, flags in plain.items()}
    for name, entry in _object(measured, "measured layers").items():
        entry = _object(entry, f"measured layer {name!r}")
        if name in layers:
            raise ArmSpecError(f"layer {name!r} is defined in both spec and measured")
        source = entry.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ArmSpecError(f"measured layer {name!r} needs a 'source'")
        layers[name] = Layer(dict(_object(entry.get("flags"), f"measured {name!r} flags")),
                             bool(entry.get("provisional", True)), source)
    return layers


def merge(names: Sequence[str], layers: Mapping[str, Layer], where: str) -> Dict[str, Any]:
    """Merge layer flags in order; an unknown layer name is an error."""
    flags: Dict[str, Any] = {}
    for name in names:
        if name not in layers:
            raise ArmSpecError(f"{where}: unknown layer {name!r}")
        flags.update(layers[name].flags)
    return flags


def describe(job: str, arm: str, names: Sequence[str], layers: Mapping[str, Layer],
             note: str = "") -> str:
    used = ", ".join(names) if names else "none (the baseline)"
    text = f"{job} arm {arm}: layers {used}"
    provisional = [f"{name} ({layers[name].source})" for name in names
                   if layers[name].provisional]
    if provisional:
        text += "; PROVISIONAL " + "; ".join(provisional)
    return text + (f"; {note}" if note else "")


def build_arm(job: str, arm: str, entry: Mapping[str, Any],
              layers: Mapping[str, Layer]) -> Dict[str, Any]:
    """One variant document, validated with the variant parser."""
    where = f"{job} arm {arm}"
    names = entry.get("layers", [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ArmSpecError(f"{where}: 'layers' must be a list of layer names")
    flags = merge(names, layers, where)  # first: it refuses unknown layer names
    document = {"name": f"{job}-{arm}",
                "description": describe(job, arm, names, layers, entry.get("note", "")),
                "flags": dict(sorted(flags.items())),
                "arms": list(entry.get("arms", ARMS))}
    try:
        parse_variant(document)
    except (VariantError, ValueError) as error:
        raise ArmSpecError(f"{where}: {error}") from error
    return document


def render(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def build_all(spec: Mapping[str, Any], measured: Mapping[str, Any]) -> Dict[str, str]:
    """``{"<job>/<arm>.json": text}`` for every arm in the spec."""
    layers = layers_from(spec, measured)
    files: Dict[str, str] = {}
    for job, arms in _object(spec.get("jobs"), "spec 'jobs'").items():
        for arm, entry in _object(arms, f"job {job!r}").items():
            document = build_arm(job, arm, _object(entry, f"{job} arm {arm}"), layers)
            files[f"{job}/{arm}.json"] = render(document)
    return files


def load(directory: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def read(name: str) -> Dict[str, Any]:
        try:
            return _object(json.loads((directory / name).read_text(encoding="utf-8")), name)
        except json.JSONDecodeError as error:
            raise ArmSpecError(f"{directory / name} is not valid JSON: {error}") from error
    return read("spec.json"), read("measured.json")


def check(directory: Path, files: Mapping[str, str]) -> List[str]:
    """Problems between the generated set and the files on disk; empty means in sync."""
    problems = []
    for relative, text in sorted(files.items()):
        path = directory / relative
        if not path.exists():
            problems.append(f"missing: {relative}")
        elif path.read_text(encoding="utf-8") != text:
            problems.append(f"differs from the spec: {relative}")
    on_disk = {str(p.relative_to(directory)) for p in directory.glob("*/*.json")}
    problems += [f"not produced by the spec: {name}" for name in sorted(on_disk - set(files))]
    return problems


def write(directory: Path, files: Mapping[str, str]) -> List[str]:
    """Write every generated file and remove stale ones; return the paths written."""
    for stale in directory.glob("*/*.json"):
        if str(stale.relative_to(directory)) not in files:
            stale.unlink()
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return sorted(files)
