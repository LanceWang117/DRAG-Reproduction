"""Fixed five-dataset DRAG pilot, with one shared model/index initialization."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/workspace/Capstone_project')
RESOURCE_REPO = ROOT / 'source_repos/Debate-Augmented-RAG'
SOURCE_REPO = ROOT / 'source_repos/Debate-Augmented-RAG-v1.0.1'
SOURCE_COMMIT = '3870c0ea3b675da63e4bc8047af07604073568f1'
NQ_RUN = ROOT / 'outputs/20261017T010415Z_drag_50q'
SPECS = [('TriviaQA', 'dev'), ('PopQA', 'test'), ('HotpotQA', 'dev'),
         ('2wiki', 'dev'), ('StrategyQA', 'train')]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, resource_repo=RESOURCE_REPO, nq_run=NQ_RUN):
    out.mkdir(parents=True, exist_ok=False)
    entries, manifest_rows = [], []
    selections = [('NQ', 'dev', nq_run / 'fixed-dataset/NQ/dev.jsonl', None)]
    selections += [(name, split, resource_repo / f'dataset/{name}/{split}.jsonl',
                    2024 + index) for index, (name, split) in enumerate(SPECS)]
    for name, split, source, seed in selections:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        selected = rows if seed is None else random.Random(seed).sample(rows, 10)
        expected = 50 if name == 'NQ' else 10
        if len(selected) != expected or len({row['id'] for row in selected}) != expected:
            raise ValueError(f'Invalid subset count or duplicate IDs: {name}')
        if any(not row.get('question') or not row.get('golden_answers') for row in selected):
            raise ValueError(f'Missing question or gold answers: {name}')
        path = out / f'fixed-datasets/{name}/{split}.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in selected))
        entries.append(dict(dataset=name, split=split, count=expected, selection_seed=seed,
                            source=str(source), source_sha256=sha(source),
                            subset=str(path), subset_sha256=sha(path),
                            ordered_ids=[row['id'] for row in selected],
                            action='reuse_completed_run' if name == 'NQ' else 'run_now'))
        for position, row in enumerate(selected):
            manifest_rows.append(dict(dataset=name, split=split, position=position,
                                      id=row['id'], question=row['question'],
                                      golden_answers=row['golden_answers']))
    fixed_manifest = out / 'question-manifest-100.jsonl'
    fixed_manifest.write_text(''.join(json.dumps(row) + '\n' for row in manifest_rows))
    manifest = dict(new_questions=50, total_fixed_questions=100, entries=entries,
                    question_manifest_sha256=sha(fixed_manifest), reused_nq_run=str(nq_run),
                    dataset_revision='bcafb8dd07d453be3cbeeeb3f78be1841bddf92c',
                    strategyqa_note='Available upstream train split; exploratory evaluation only, no training. '
                                    'Not a held-out test or original-paper split claim.')
    write_json(out / 'sample-manifest.json', manifest)
    return manifest


class TimedDataset:
    """Time only the first inference traversal; evaluation uses the same items."""
    def __init__(self, dataset, out, name, split):
        self.dataset, self.out, self.name, self.split = dataset, out, name, split
        self.finished = False
        self.elapsed = []

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __iter__(self):
        if self.finished:
            yield from self.dataset
            return
        try:
            for index, item in enumerate(self.dataset):
                started = time.monotonic()
                yield item
                elapsed = time.monotonic() - started
                self.elapsed.append(elapsed)
                item.update_output('inference_elapsed_seconds', elapsed)
                record = dict(dataset=self.name, split=self.split, position=index,
                              id=item.id, elapsed_seconds=elapsed, item=item.to_dict())
                with (self.out / 'question-checkpoints.jsonl').open('a') as log:
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                write_json(self.out / 'progress.json', dict(
                    dataset=self.name, split=self.split, completed=index + 1,
                    total=len(self.dataset), last_id=item.id))
        finally:
            self.finished = True


def worker(out):
    os.environ.update(HF_HOME=str(ROOT / 'cache/huggingface'),
                      HF_HUB_ENABLE_HF_TRANSFER='0', TMPDIR=str(ROOT / 'cache/tmp'),
                      OMP_NUM_THREADS='12', MKL_NUM_THREADS='12',
                      TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(SOURCE_REPO))
    started = time.monotonic()
    from flashrag.config import Config
    from flashrag.dataset import Dataset
    from flashrag.utils import get_generator, get_retriever
    from model.drag import DebateAugmentedRAG

    def configuration(name, split):
        return Config(str(SOURCE_REPO / 'config/base_config.yaml'), {
            'model2path': {'e5': str(ROOT / 'models/e5-base-v2'),
                           'llama3-8B-instruct': str(ROOT / 'models/Llama-3.1-8B-Instruct')},
            'framework': 'vllm', 'generator_model': 'llama3-8B-instruct',
            'gpu_id': '0', 'faiss_gpu': False, 'seed': 2024,
            'index_path': str(RESOURCE_REPO / 'wiki_corpus/e5_flat_inner.index'),
            'corpus_path': str(RESOURCE_REPO / 'wiki_corpus/wiki18_100w.jsonl'),
            'data_dir': str(out / 'fixed-datasets'), 'dataset_name': name,
            'split': [split], 'random_sample': False, 'test_sample_num': 10,
            'save_dir': str(out / 'official-results'), 'save_note': 'DRAG_v1.0.1_extra10',
            'save_intermediate_data': True, 'save_metric_score': True,
            'metrics': ['em', 'f1', 'acc', 'precision', 'recall'],
        })

    base_config = configuration(*SPECS[0])
    print('SHARED_MODEL_INDEX_INIT_START', flush=True)
    generator = get_generator(base_config)
    retriever = get_retriever(base_config)
    initialization = time.monotonic() - started
    print('SHARED_MODEL_INDEX_INIT_COMPLETE', initialization, flush=True)
    results = []
    for name, split in SPECS:
        try:
            config = configuration(name, split)
            dataset = Dataset(config=config, dataset_path=str(out / f'fixed-datasets/{name}/{split}.jsonl'))
            tracked = TimedDataset(dataset, out, name, split)
            pipeline = DebateAugmentedRAG(config, generator=generator, retriever=retriever,
                                          max_query_debate_rounds=3, max_answer_debate_rounds=3)
            inference_start = time.monotonic()
            pipeline.run(tracked)
            elapsed = time.monotonic() - inference_start
            metric_path = Path(config['save_dir']) / 'metric_score.txt'
            metrics = {line.partition(':')[0]: float(line.partition(':')[2])
                       for line in metric_path.read_text().splitlines() if ':' in line}
            if len(tracked.elapsed) != 10 or len(metrics) != 5:
                raise RuntimeError('Question timing or metrics incomplete')
            results.append(dict(dataset=name, split=split, status='complete', questions=10,
                                metrics=metrics, question_inference_seconds=sum(tracked.elapsed),
                                average_question_seconds=sum(tracked.elapsed) / 10,
                                dataset_wall_seconds_including_evaluation=elapsed,
                                output_dir=config['save_dir']))
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            results.append(dict(dataset=name, split=split, status='failed', error=error))
        write_json(out / 'dataset-results.json', results)
    write_json(out / 'experiment-summary.json', dict(
        version='v1.0.1', source_commit=SOURCE_COMMIT, new_questions_requested=50,
        completed_questions=sum(r.get('questions', 0) for r in results),
        shared_initialization_including_imports_seconds=initialization,
        worker_wall_seconds=time.monotonic() - started, datasets=results,
        reused_nq_run=str(NQ_RUN), note='NQ 50 reused, not executed again. See per-dataset splits.'))
    return 0 if all(r['status'] == 'complete' for r in results) else 1


def supervise(out):
    import psutil
    from cloud_run_smoke import gpu_sample, read_cgroup_usage
    started = time.monotonic()
    timed_out, peak_rss = False, 0
    with (out / 'console.log').open('w') as log:
        process = subprocess.Popen([sys.executable, __file__, 'worker', '--out', str(out)],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= 7200:
                    timed_out = True
                    break
                rss = 0
                try:
                    parent = psutil.Process(process.pid)
                    for child in [parent] + parent.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                peak_rss = max(peak_rss, rss)
                try:
                    gpu = gpu_sample()
                except (subprocess.TimeoutExpired, OSError):
                    gpu = None
                sample = dict(elapsed_seconds=elapsed, process_tree_sum_rss_bytes=rss,
                              pod_cgroup_memory_bytes=read_cgroup_usage(),
                              gpu_mib_percent_watts=gpu)
                with (out / 'resources.jsonl').open('a') as record:
                    record.write(json.dumps(sample) + '\n')
                time.sleep(15)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    write_json(out / 'run-summary.json', dict(
        source_commit=SOURCE_COMMIT, version='v1.0.1', method='DRAG',
        new_questions_requested=50, return_code=process.returncode,
        timed_out=timed_out, elapsed_seconds=time.monotonic() - started,
        peak_process_tree_sum_rss_bytes=peak_rss,
        resource_note='RSS sum can double-count shared pages; cgroup includes file cache.'))
    return process.returncode or int(timed_out)


def launch():
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid != os.getpid() and any(Path(a).name in (
            'cloud_run_smoke.py', 'cloud_multiset_batch.py', 'main.py'
        ) for a in process.info['cmdline'] or []):
            raise RuntimeError(f'Existing experiment PID {process.pid}; refusing duplicate')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE_REPO, text=True).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError('Unexpected DRAG source version')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=SOURCE_REPO, check=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'outputs' / f'{stamp}_drag_extra50_v1.0.1'
    manifest = prepare(out)
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    with (out / 'supervisor.log').open('w') as log:
        process = subprocess.Popen([sys.executable, __file__, 'supervise', '--out', str(out)],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    record = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
                  output_dir=str(out), version='v1.0.1', source_commit=commit,
                  runner_sha256=sha(Path(__file__)), new_questions=manifest['new_questions'],
                  total_fixed_questions=100, datasets=[dict(dataset=n, split=s, count=10) for n, s in SPECS],
                  max_runtime_seconds=7200, note='Timeout ends experiment, NOT Pod billing.')
    write_json(out / 'launch.json', record)
    write_json(ROOT / 'outputs/extra50-latest-launch.json', record)
    time.sleep(2)
    record['alive_after_launch'] = process.poll() is None
    print(json.dumps(record, indent=2), flush=True)
    if process.poll() is not None:
        raise RuntimeError(f'Experiment supervisor exited early: {process.returncode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'supervise', 'worker'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode != 'launch' and args.out is None:
        parser.error('--out is required')
    if args.mode == 'launch':
        launch()
    else:
        sys.exit(worker(args.out) if args.mode == 'worker' else supervise(args.out))
