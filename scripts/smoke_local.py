"""Run 3-5 dev questions end to end with a real small model (local Ollama), then summarise.

This catches prompt, parsing and stage-flow bugs on real model text before the A100 stage.
It is never a source of results: the default model (llama3.2:3b) is a different family and
size than the A100's Llama-3.1-8B BF16, so nothing it prints is comparable, and its output
under new-runs/local-smoke/ is ignored by git and never committed.

The generator calls the Ollama server's OpenAI-compatible API; the retriever replays the
stored passages from results/retrieval-200.jsonl. If no server answers, the script starts
`ollama serve` itself and stops it (and unloads the model) at the end, even on failure; a
server that was already running is left alone. It refuses to start with under 3 GiB of
free VRAM, 4 GiB of free RAM or 1 GiB of free disk, and pulls a missing model only if it is
a tag of 4B parameters or fewer and 10 GiB of disk are free.

Exit status: 0 all variants passed, 1 a variant failed, 2 preflight or server refused.
"""
import argparse
import signal
import sys
from pathlib import Path

from run_evaluation import MANIFEST, PROJECT, select_questions
from run_variant import DATASETS
sys.path.insert(0, str(PROJECT / 'improvements'))
from drag_plus.runner.local_generator import DEFAULT_MODEL  # noqa: E402
from drag_plus.runner.local_preflight import PreflightError, read_hardware  # noqa: E402
from drag_plus.runner.local_server import DEFAULT_HOST, OllamaServer, ServerError  # noqa: E402
from drag_plus.runner.local_smoke import (MAX_QUESTIONS, SmokeJob, run_smoke,  # noqa: E402
                                          run_variant_once)
from drag_plus.runner.smoke_summary import format_summary  # noqa: E402
from drag_plus.runner.variant import load_variant  # noqa: E402

DEFAULT_VARIANT = PROJECT / 'improvements/configs/local-smoke.json'


def question_count(text):
    value = int(text)
    if not 1 <= value <= MAX_QUESTIONS:
        raise argparse.ArgumentTypeError(f'must be between 1 and {MAX_QUESTIONS}')
    return value


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=question_count, default=3, help='Dev questions, 1-5')
    parser.add_argument('--variant', type=Path, nargs='+', default=[DEFAULT_VARIANT],
                        help='Variant JSON files; each runs on the same questions')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Ollama model tag')
    parser.add_argument('--host', default=DEFAULT_HOST, help='Ollama host:port')
    parser.add_argument('--dataset', choices=DATASETS, help='Only questions from this dataset')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--replay', type=Path, default=PROJECT / 'results/retrieval-200.jsonl')
    parser.add_argument('--output', type=Path, default=PROJECT / 'new-runs/local-smoke')
    parser.add_argument('--no-pull', action='store_true', help='Never download a missing model')
    return parser.parse_args(argv)


def default_server(host, log_path):
    return OllamaServer(host, log_path=log_path)


def main(argv=None, *, hardware=read_hardware, server_factory=default_server):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    variants = [load_variant(path) for path in args.variant]
    rows = select_questions(args.manifest, args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)
    job = SmokeJob(args.model, rows, variants, args.output, args.replay, args.manifest,
                   allow_pull=not args.no_pull)
    server = server_factory(args.host, args.output / 'ollama-serve.log')
    try:
        results = run_smoke(job, server, hardware(args.output),
                            lambda job, variant: run_variant_once(job, variant, f'{server.url}/v1'))
    except (PreflightError, ServerError) as error:
        print(f'Smoke run refused: {error}', file=sys.stderr)
        return 2
    print(format_summary(results, f'ollama:{args.model}', len(rows)))
    return 0 if all(result['ok'] for result in results) else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))  # run the finally blocks
    sys.exit(main())
