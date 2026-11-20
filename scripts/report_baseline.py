"""Print the recorded baseline comparison, recomputed from the stored answers.

Usage: python scripts/report_baseline.py [--pairs results/paired-200.json]
"""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "improvements"))
from drag_plus.analysis.report import load_pairs, render  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, default=PROJECT / "results" / "paired-200.json")
    args = parser.parse_args(argv)
    print(render(load_pairs(args.pairs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
