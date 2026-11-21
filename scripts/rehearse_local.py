"""Rehearse an E-job's arms on the PC's local model before booking the A100.

Runs each variant on 6-30 dev questions (the first N/6 per dataset, a prefix of E00's
subset) through the variant runner with the local Ollama backend and the stored
passages, then prints a structural report: model and retrieval calls per question,
rounds, tokens, finish reasons, Judge statuses, parse failures and errors. It is never a
source of results: no accuracy or latency on the small local model is comparable with
the A100 runs, and its output under new-runs/local-rehearsal/ is never committed.

Arms that need A100-only machinery (hybrid retrieval, BM25 index, reranker, GPU FAISS,
prefix caching, quantisation, speculative decoding) are skipped with their reasons. Only
the dev manifest is accepted. ``--resume RUN_DIR`` continues an interrupted rehearsal.

Exit status: 0 every run arm passed, 1 an arm failed or none could run,
2 refused (guard, preflight or server).
"""
import argparse
import datetime
import signal
import sys
from pathlib import Path

from run_evaluation import MANIFEST, PROJECT, select_questions
sys.path.insert(0, str(PROJECT / 'improvements'))
from drag_plus.rehearsal.plan import RehearsalError, check_manifest, plan_arms, select_rows  # noqa: E402
from drag_plus.rehearsal.report_text import format_report, overall_ok  # noqa: E402
from drag_plus.rehearsal.session import estimate_minutes, run_rehearsal  # noqa: E402
from drag_plus.runner.local_backend import build_local_backend  # noqa: E402
from drag_plus.runner.local_generator import DEFAULT_MODEL  # noqa: E402
from drag_plus.runner.local_preflight import PreflightError, read_hardware  # noqa: E402
from drag_plus.runner.local_server import DEFAULT_HOST, OllamaServer, ServerError  # noqa: E402
from drag_plus.runner.local_smoke import SmokeJob  # noqa: E402
from drag_plus.runner.resume import ResumeError  # noqa: E402
from drag_plus.runner.variant import load_variant  # noqa: E402

OUTPUT = PROJECT / 'new-runs/local-rehearsal'


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', type=Path, nargs='+', required=True,
                        help='Variant JSON files (the E-job arms), in report order')
    parser.add_argument('--questions', type=int, default=12, help='Dev questions, a multiple of 6 in 6-30')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Ollama model tag')
    parser.add_argument('--host', default=DEFAULT_HOST, help='Ollama host:port')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--replay', type=Path, default=PROJECT / 'results/retrieval-200.jsonl')
    parser.add_argument('--resume', type=Path, metavar='RUN_DIR', help='Continue this rehearsal')
    parser.add_argument('--no-pull', action='store_true', help='Never download a missing model')
    return parser.parse_args(argv)


def _prepare(args):
    check_manifest(args.manifest, MANIFEST)
    rows = select_rows(select_questions(args.manifest), args.questions)
    variants = [load_variant(path) for path in args.variant]
    plans = plan_arms(variants)
    runnable = [v for v, plan in zip(variants, plans) if plan.runnable]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = args.resume or OUTPUT / stamp
    return rows, runnable, plans, root


def default_server(host, log_path):
    return OllamaServer(host, log_path=log_path)


def _backend_factory(server):
    return lambda job: build_local_backend(model=job.model, base_url=f'{server.url}/v1',
                                           retrieval_path=job.replay, questions_path=job.questions_path)


def main(argv=None, *, hardware=read_hardware, server_factory=default_server, backend_factory=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        rows, runnable, plans, root = _prepare(args)
    except RehearsalError as error:
        print(f'Rehearsal refused: {error}', file=sys.stderr)
        return 2
    reports = []
    if runnable:
        print(f'Estimated local time: about {estimate_minutes(runnable, len(rows)):.0f} min; '
              f'output {root}', flush=True)
        root.mkdir(parents=True, exist_ok=True)
        job = SmokeJob(args.model, rows, runnable, root, args.replay, args.manifest,
                       allow_pull=not args.no_pull)
        server = server_factory(args.host, root / 'ollama-serve.log')
        try:
            reports = run_rehearsal(job, server, hardware(root),
                                    (backend_factory or _backend_factory)(server))
        except (PreflightError, ServerError, ResumeError) as error:
            print(f'Rehearsal refused: {error}', file=sys.stderr)
            return 2
    print(format_report(reports, plans, f'ollama:{args.model}', len(rows)))
    return 0 if overall_ok(reports) else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))  # run the finally blocks
    sys.exit(main())
