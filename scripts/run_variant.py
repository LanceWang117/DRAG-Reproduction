"""Run a DRAG variant (drag_plus and Naive arms) with per-call timing and resume.

Unlike scripts/run_evaluation.py, which reproduces the recorded baseline, this runner takes
a variant file (flags + arms) and writes stage-timings.jsonl, transcripts.jsonl and
run-manifest.json into new-runs/<UTC>-<variant>/. ``--backend fake`` runs the whole path
offline; ``--backend flashrag`` needs the A100 stack and has not been run on the laptop.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from prepare_source import build_runtime
from run_evaluation import MANIFEST, PROJECT, resolve_paths, select_questions

DATASETS = ['NQ', 'TriviaQA', 'PopQA', '2wiki', 'HotpotQA', 'StrategyQA']
sys.path.insert(0, str(PROJECT / 'improvements'))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', type=Path, required=True, help='Variant JSON file')
    parser.add_argument('--backend', choices=['fake', 'flashrag'], default='flashrag')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config.example.json')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--limit', type=int, help='First N matching questions, without resampling')
    parser.add_argument('--output', type=Path, default=PROJECT / 'new-runs')
    parser.add_argument('--profile', action='store_true', help='Set the profile_calls flag')
    parser.add_argument('--interleave', action='store_true', help='Alternate arms per question')
    parser.add_argument('--resume', type=Path, metavar='RUN_DIR', help='Continue an earlier run')
    parser.add_argument('--validate-only', action='store_true',
                        help='Check inputs without importing heavy libraries')
    return parser.parse_args(argv)


def validate(args):
    """Check the variant, the questions and (for flashrag) the config file."""
    from drag_plus.runner.variant import load_variant
    variant = load_variant(args.variant)
    rows = select_questions(args.manifest, args.dataset, args.limit)
    if args.backend == 'flashrag':
        resolve_paths(json.loads(args.config.read_text()), args.config.parent)
    return variant, rows


def build_backend(args, rows, out_dir):
    from drag_plus.runner.backends import build_fake_backend, build_flashrag_backend
    if args.backend == 'fake':
        return build_fake_backend()
    overrides = resolve_paths(json.loads(args.config.read_text()), args.config.parent)
    for value in [*overrides['model2path'].values(), overrides['index_path'], overrides['corpus_path']]:
        if not Path(value).exists():
            raise FileNotFoundError(f'Resource not found: {value}; see DATA_DOWNLOADS.md')
    os.environ.setdefault('OMP_NUM_THREADS', '12')
    os.environ.setdefault('MKL_NUM_THREADS', '12')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    source = build_runtime()
    sys.path.insert(0, str(source))
    return build_flashrag_backend(source, overrides, rows, out_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    variant, rows = validate(args)
    print('Variant:', variant.name, '| arms:', ', '.join(variant.arms), '| questions:', len(rows))
    if args.validate_only:
        print('Input validation passed; no model loaded.')
        return 0
    from drag_plus.runner.run import RunRequest, default_run_dir, run_variant
    out_dir = args.resume or default_run_dir(args.output, variant.name)
    request = RunRequest(
        rows=rows, variant=variant, backend=build_backend(args, rows, out_dir), out_dir=out_dir,
        profile=args.profile, interleave=args.interleave, resume=args.resume is not None,
        command=['run_variant.py', *argv], progress=lambda text: print(text, flush=True))
    result = run_variant(request)
    print('Saved:', out_dir, '| exit status:', result['exit_status'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
