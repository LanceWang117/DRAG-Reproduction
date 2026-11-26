"""Write (or check) every E-job arm file from improvements/configs/arms/spec.json.

    python scripts/make_arms.py            # regenerate improvements/configs/arms/<job>/<arm>.json
    python scripts/make_arms.py --check    # exit 1 if a committed file differs from the spec

After E01 promotes a configuration (or E00/E04 measure their values), edit
improvements/configs/arms/measured.json, set "provisional": false, and rerun this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "improvements"))

from drag_plus.runner.arms import ArmSpecError, build_all, check, load, write  # noqa: E402

DEFAULT_DIR = PROJECT / "improvements/configs/arms"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Arm spec folder")
    parser.add_argument("--check", action="store_true", help="Compare, never write")
    args = parser.parse_args(argv)
    try:
        files = build_all(*load(args.dir))
    except ArmSpecError as error:
        print(f"make_arms: {error}", file=sys.stderr)
        return 2
    if args.check:
        problems = check(args.dir, files)
        for problem in problems:
            print(f"make_arms: {problem}", file=sys.stderr)
        if problems:
            print("make_arms: run `python scripts/make_arms.py` and commit the result",
                  file=sys.stderr)
            return 1
        print(f"make_arms: {len(files)} arm files match the spec")
        return 0
    written = write(args.dir, files)
    print(f"make_arms: wrote {len(written)} arm files under {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
