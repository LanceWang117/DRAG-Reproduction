#!/usr/bin/env python3
"""Run the unmodified official entry point and sample resource use."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path('/workspace/Capstone_project')
REPO = ROOT / 'source_repos/Debate-Augmented-RAG'


def read_cgroup_usage():
    for path in ('/sys/fs/cgroup/memory.current',
                 '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            pass
    return None


def gpu_sample():
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['Naive RAG', 'DRAG'], required=True)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--resource-interval-seconds', type=int, default=5)
    parser.add_argument('--repo-path', type=Path, default=REPO)
    parser.add_argument('--paired-baseline-dir', type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples <= 50:
        parser.error('This verification runner is limited to 1-50 questions')
    if args.resource_interval_seconds < 5:
        parser.error('Resource monitoring interval must be at least 5 seconds')
    source_repo = args.repo_path.resolve()
    if not (source_repo / 'main.py').is_file():
        parser.error('Source repository must contain main.py')
    for required in (
        ROOT / 'models/Llama-3.1-8B-Instruct/config.json',
        ROOT / 'models/e5-base-v2/config.json',
        REPO / 'wiki_corpus/wiki18_100w.jsonl',
        REPO / 'wiki_corpus/e5_flat_inner.index',
        REPO / 'dataset/NQ/dev.jsonl',
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    label = args.method.lower().replace(' ', '_')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'outputs' / f'{stamp}_{label}_{args.samples}q'
    out.mkdir(parents=True, exist_ok=False)
    config = {
        'model2path': {
            'e5': str(ROOT / 'models/e5-base-v2'),
            'llama3-8B-instruct': str(ROOT / 'models/Llama-3.1-8B-Instruct'),
        },
        'framework': 'vllm',
        'generator_model': 'llama3-8B-instruct',
        'gpu_id': '0',
        'faiss_gpu': False,
        'index_path': str(REPO / 'wiki_corpus/e5_flat_inner.index'),
        'corpus_path': str(REPO / 'wiki_corpus/wiki18_100w.jsonl'),
        'data_dir': str(REPO / 'dataset'),
        'dataset_name': 'NQ',
        'split': ['dev'],
        'seed': 2024,
        'random_sample': True,
        'save_intermediate_data': True,
        'metrics': ['em', 'f1', 'acc', 'precision', 'recall'],
    }
    if args.paired_baseline_dir is not None:
        baseline_files = list(args.paired_baseline_dir.rglob('intermediate_data.json'))
        if len(baseline_files) != 1:
            raise ValueError('Paired baseline must contain exactly one saved result file')
        baseline = json.loads(baseline_files[0].read_text())
        if len(baseline) != args.samples:
            raise ValueError('Paired baseline question count differs from requested samples')
        original_path = REPO / 'dataset/NQ/dev.jsonl'
        original = [json.loads(line) for line in original_path.read_text().splitlines() if line.strip()]
        by_id = {row['id']: row for row in original}
        selected = [by_id[item['id']] for item in baseline]
        if len({row['id'] for row in selected}) != args.samples:
            raise ValueError('Paired baseline contains duplicate question IDs')
        for row, item in zip(selected, baseline):
            for field in ('question', 'golden_answers'):
                if row[field] != item[field]:
                    raise ValueError(f'Paired question mismatch: {row["id"]} {field}')
        subset_path = out / 'fixed-dataset/NQ/dev.jsonl'
        subset_path.parent.mkdir(parents=True)
        subset_path.write_text(''.join(json.dumps(row) + '\n' for row in selected))
        config['data_dir'] = str(out / 'fixed-dataset')
        config['random_sample'] = False
        manifest = dict(
            paired_baseline=str(baseline_files[0]), seed=config['seed'],
            ordered_question_ids=[row['id'] for row in selected],
            source_dataset_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
            fixed_subset_sha256=hashlib.sha256(subset_path.read_bytes()).hexdigest(),
            note='Same ordered IDs, questions and gold answers as baseline; no resampling.',
        )
        (out / 'sample-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    config_file = out / 'config-override.json'
    config_file.write_text(json.dumps(config, indent=2) + '\n')
    command = [
        sys.executable, 'main.py', '--method_name', args.method,
        '--config_path', str(config_file), '--generator_model', 'llama3-8B-instruct',
        '--dataset_name', 'NQ', '--test_sample_num', str(args.samples),
        '--gpu_id', '0', '--save_dir', str(out / 'official-results'),
        '--max_query_debate_rounds', '3', '--max_answer_debate_rounds', '3',
    ]
    environment = os.environ.copy()
    environment.update({
        'HF_HOME': str(ROOT / 'cache/huggingface'),
        'HF_HUB_ENABLE_HF_TRANSFER': '0',
        'TMPDIR': str(ROOT / 'cache/tmp'),
        'OMP_NUM_THREADS': '12',
        'MKL_NUM_THREADS': '12',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONUNBUFFERED': '1',
    })
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=source_repo, text=True).strip()
    started = time.monotonic()
    samples = []
    timed_out = False
    with (out / 'console.log').open('w') as log:
        process = subprocess.Popen(command, cwd=source_repo, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        tracked = {}
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout_seconds:
                    timed_out = True
                    break
                rss, cpu = 0, 0.0
                try:
                    parent = psutil.Process(process.pid)
                    for child in [parent] + parent.children(recursive=True):
                        try:
                            key = (child.pid, child.create_time())
                            if key not in tracked:
                                tracked[key] = child
                            tracked_child = tracked[key]
                            rss += tracked_child.memory_info().rss
                            cpu += tracked_child.cpu_percent()
                        except psutil.Error:
                            continue
                except psutil.Error:
                    pass
                sample = dict(elapsed_seconds=elapsed,
                              process_tree_sum_rss_bytes=rss,
                              process_tree_cpu_percent=cpu,
                              pod_cgroup_memory_bytes=read_cgroup_usage(),
                              gpu_mib_percent_watts=gpu_sample())
                samples.append(sample)
                with (out / 'resources.jsonl').open('a') as record:
                    record.write(json.dumps(sample) + '\n')
                time.sleep(args.resource_interval_seconds)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    summary = dict(
        method=args.method, question_count=args.samples, source_commit=commit,
        source_repo=str(source_repo), paired_baseline_dir=str(args.paired_baseline_dir) if args.paired_baseline_dir else None,
        command=command, elapsed_seconds=time.monotonic() - started,
        return_code=process.returncode, timed_out=timed_out,
        peak_process_tree_sum_rss_bytes=max(
            (s['process_tree_sum_rss_bytes'] for s in samples), default=0),
        resource_note='RSS sum can double-count shared pages; cgroup includes cache.',
        outputs=str(out),
    )
    (out / 'run-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    sys.exit(1 if timed_out else process.returncode)


if __name__ == '__main__':
    main()
