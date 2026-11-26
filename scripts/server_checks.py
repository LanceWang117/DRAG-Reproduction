"""List the server-only checks, or run the automatic ones on the A100 (E00, J26).

    python scripts/server_checks.py --list [--tier auto|smoke|measure] [--owner E00]
    python scripts/server_checks.py --run --config config.local.json        # on the Pod
    python scripts/server_checks.py --run --backend fake                    # PC: runner only

The registry is improvements/configs/server-checks.json. --run builds the same backend
as run_variant.py (default variant: configs/arms/E02b/D.json, which wraps the log-prob
scorer), runs each automatic probe once and writes server-checks.json. Exit status: 0
when nothing failed or errored, 1 otherwise, 2 for bad input.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_variant  # noqa: E402  (puts improvements/ on sys.path)
from run_evaluation import PROJECT, resolve_paths  # noqa: E402

from drag_plus.server.probes import ProbeContext  # noqa: E402
from drag_plus.server.registry import (TIERS, RegistryError, format_list, load_checks,  # noqa: E402
                                       passed, run_auto, select, summarise)

REGISTRY = PROJECT / "improvements/configs/server-checks.json"
DEFAULT_VARIANT = PROJECT / "improvements/configs/arms/E02b/D.json"
STORED_COUNT = 3


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Print the checklist as Markdown")
    mode.add_argument("--run", action="store_true", help="Run the automatic probes")
    parser.add_argument("--tier", choices=TIERS)
    parser.add_argument("--owner", help="Only checks owned by this E-job, e.g. E00")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--backend", default="flashrag", choices=("flashrag", "fake", "local"))
    parser.add_argument("--variant", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.example.json")
    parser.add_argument("--manifest", type=Path, default=run_variant.MANIFEST)
    parser.add_argument("--retrieval", type=Path, default=PROJECT / "results/retrieval-200.jsonl")
    parser.add_argument("--out", type=Path, help="Report folder (default new-runs/server-checks/<stamp>)")
    return parser.parse_args(argv)


def load_stored(manifest: Path, retrieval: Path, count: int = STORED_COUNT):
    """``(question, naive ids, naive docs)`` for the first ``count`` stored NQ rows."""
    questions = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            questions[(row["dataset"], row["id"])] = row["question"]
    stored = []
    for line in retrieval.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else None
        if row and row["dataset"] == "NQ" and (row["dataset"], row["id"]) in questions:
            docs = row["naive"]
            stored.append((questions[(row["dataset"], row["id"])], [d["id"] for d in docs], docs))
        if len(stored) == count:
            break
    return stored


def build_context(args, out_dir: Path) -> ProbeContext:
    rv_args = run_variant.parse_args(["--variant", str(args.variant), "--backend", args.backend,
                                      "--config", str(args.config), "--manifest", str(args.manifest),
                                      "--limit", "1"])
    variant, rows = run_variant.validate(rv_args)
    backend = run_variant.build_backend(rv_args, variant.flags, rows, out_dir)
    index_path, gpu_reader = None, None
    if args.backend == "flashrag":
        import preflight_gpu
        index_path = resolve_paths(json.loads(args.config.read_text()), args.config.parent)["index_path"]
        gpu_reader = preflight_gpu.read_gpu_total_gib
    return ProbeContext(backend=args.backend, generator=backend.generator,
                        retriever=backend.retriever, make_template=backend.make_template,
                        flags=variant.flags, stored=load_stored(args.manifest, args.retrieval),
                        index_path=index_path, read_gpu_gib=gpu_reader)


def write_report(out_dir: Path, args, results) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "backend": args.backend, "variant": str(args.variant),
              "summary": summarise(results), "passed": passed(results), "checks": results}
    path = out_dir / "server-checks.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def run(args) -> int:
    checks = load_checks(args.registry)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or PROJECT / "new-runs/server-checks" / stamp
    results = run_auto(select(checks, owner=args.owner), build_context(args, out_dir))
    for result in results:
        print(f"{result['status']:>7}  {result['id']:<6} {result['probe']}: {result['detail'][:120]}")
    print("summary:", summarise(results), "| report:", write_report(out_dir, args, results))
    return 0 if passed(results) else 1


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.list:
            print(format_list(select(load_checks(args.registry), args.tier, args.owner)))
            return 0
        return run(args)
    except (RegistryError, FileNotFoundError, ValueError) as error:
        print(f"server_checks: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
