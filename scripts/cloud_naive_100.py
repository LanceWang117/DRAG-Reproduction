"""Official Naive RAG on the exact fixed 100-question DRAG pilot inputs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from cloud_multiset_batch import ROOT, RESOURCE_REPO, sha, supervise, write_json

SOURCE_COMMIT = 'c035ee1b7e6bc9fa67d5a104df1c2175a8300666'
PAIRED = ROOT / 'outputs/20261017T015401Z_drag_extra50_v1.0.1'


def prepare(out, paired=PAIRED):
    source_manifest = json.loads((paired / 'sample-manifest.json').read_text())
    canonical = paired / 'question-manifest-100.jsonl'
    if sha(canonical) != source_manifest['question_manifest_sha256']:
        raise ValueError('Canonical question manifest checksum differs')
    canonical_rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    actual_rows = []
    for entry in source_manifest['entries']:
        name, split = entry['dataset'], entry['split']
        source = paired / 'fixed-datasets' / name / (split + '.jsonl')
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if sha(source) != entry['subset_sha256'] or [r['id'] for r in rows] != entry['ordered_ids']:
            raise ValueError(f'Fixed subset checksum or ID mismatch: {name}')
        for index, row in enumerate(rows):
            actual_rows.append(dict(dataset=name, split=split, position=index,
                                    id=row['id'], question=row['question'],
                                    golden_answers=row['golden_answers']))
    if len(actual_rows) != 100 or actual_rows != canonical_rows:
        raise ValueError('Paired ordered questions/gold do not match canonical 100')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(paired / 'fixed-datasets', out / 'fixed-datasets')
    shutil.copy2(canonical, out / canonical.name)
    manifest = dict(source_manifest, method='Naive RAG', questions_to_run=100, new_questions=100,
                    paired_drag_inputs=str(paired), resampled=False)
    manifest['paired_drag_nq_run'] = manifest.pop('reused_nq_run')
    manifest['entries'] = [dict(entry, action='run_now', paired_drag_action=entry['action'])
                           for entry in source_manifest['entries']]
    write_json(out / 'sample-manifest.json', manifest)
    return manifest


def worker(out):
    os.environ.update(HF_HOME=str(ROOT / 'cache/huggingface'), HF_HUB_ENABLE_HF_TRANSFER='0',
                      TMPDIR=str(ROOT / 'cache/tmp'), OMP_NUM_THREADS='12',
                      MKL_NUM_THREADS='12', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(RESOURCE_REPO))
    started = time.monotonic()
    from flashrag.config import Config
    from flashrag.dataset import Dataset
    from flashrag.pipeline import SequentialPipeline
    from flashrag.utils import get_generator, get_retriever
    from model.prompt import standard_rag_prompt_template
    from model.utils import single_agent_pred_parse

    manifest = json.loads((out / 'sample-manifest.json').read_text())
    entries = manifest['entries']

    def configuration(entry):
        return Config(str(RESOURCE_REPO / 'config/base_config.yaml'), {
            'model2path': {'e5': str(ROOT / 'models/e5-base-v2'),
                           'llama3-8B-instruct': str(ROOT / 'models/Llama-3.1-8B-Instruct')},
            'framework': 'vllm', 'generator_model': 'llama3-8B-instruct',
            'gpu_id': '0', 'faiss_gpu': False, 'seed': 2024,
            'index_path': str(RESOURCE_REPO / 'wiki_corpus/e5_flat_inner.index'),
            'corpus_path': str(RESOURCE_REPO / 'wiki_corpus/wiki18_100w.jsonl'),
            'dataset_name': entry['dataset'], 'split': [entry['split']],
            'data_dir': str(out / 'fixed-datasets'), 'random_sample': False,
            'test_sample_num': entry['count'], 'save_dir': str(out / 'official-results'),
            'save_note': 'Naive_RAG_fixed100_serial', 'save_intermediate_data': True,
            'save_metric_score': True, 'metrics': ['em', 'f1', 'acc', 'precision', 'recall'],
        })

    config = configuration(entries[0])
    print('SHARED_MODEL_INDEX_INIT_START', flush=True)
    generator = get_generator(config)
    retriever = get_retriever(config)
    initialization = time.monotonic() - started
    print('SHARED_MODEL_INDEX_INIT_COMPLETE', initialization, flush=True)
    results, completed = [], 0
    for entry in entries:
        name, split = entry['dataset'], entry['split']
        try:
            config = configuration(entry)
            dataset = Dataset(config=config, dataset_path=str(out / 'fixed-datasets' / name / (split + '.jsonl')))
            pipeline = SequentialPipeline(config, standard_rag_prompt_template[name](config),
                                          generator=generator, retriever=retriever)
            times = []
            dataset_start = time.monotonic()
            for position, item in enumerate(dataset):
                single = Dataset(config=config, data=[item])
                before = time.monotonic()
                pipeline.run(single, do_eval=False, pred_process_fun=single_agent_pred_parse)
                elapsed = time.monotonic() - before
                if not item.pred.strip() or len(item.retrieval_result) != config['retrieval_topk']:
                    raise RuntimeError(f'Empty answer or incomplete retrieval: {name}/{item.id}')
                item.update_output('inference_elapsed_seconds', elapsed)
                times.append(elapsed)
                if completed == 0:
                    smoke_metrics = pipeline.evaluator.evaluate(single)
                    if len(smoke_metrics) != 5:
                        raise RuntimeError('Smoke evaluation incomplete; refusing full batch')
                    write_json(out / 'smoke-check.json', dict(
                        status='passed', dataset=name, split=split, id=item.id,
                        elapsed_seconds=elapsed, metrics=smoke_metrics, item=item.to_dict(),
                        note='Functional check, not an accuracy threshold; included as question 1/100.'))
                    print('SMOKE_PASSED_CONTINUING_FIXED_100', item.id, flush=True)
                record = dict(dataset=name, split=split, position=position,
                              id=item.id, elapsed_seconds=elapsed, item=item.to_dict())
                with (out / 'question-checkpoints.jsonl').open('a') as log:
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                completed += 1
                write_json(out / 'progress.json', dict(
                    completed=completed, total=100, dataset=name, split=split,
                    dataset_completed=position + 1, dataset_total=len(dataset), last_id=item.id))
                print('QUESTION_COMPLETE', completed, name, item.id, round(elapsed, 3), flush=True)
            metrics = pipeline.evaluator.evaluate(dataset)
            if len(times) != entry['count'] or len(metrics) != 5:
                raise RuntimeError('Incomplete dataset results')
            results.append(dict(dataset=name, split=split, status='complete', questions=len(times),
                                metrics=metrics, question_inference_seconds=sum(times),
                                average_question_seconds=sum(times) / len(times),
                                dataset_wall_seconds_including_evaluation=time.monotonic() - dataset_start,
                                output_dir=config['save_dir']))
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            results.append(dict(dataset=name, split=split, status='failed', error=error))
            write_json(out / 'dataset-results.json', results)
            # Do not continue a batch whose initial functional check failed.
            if not (out / 'smoke-check.json').exists():
                raise
        write_json(out / 'dataset-results.json', results)
    write_json(out / 'experiment-summary.json', dict(
        method='Naive RAG', source_commit=SOURCE_COMMIT, completed_questions=completed,
        shared_initialization_including_imports_seconds=initialization,
        worker_wall_seconds=time.monotonic() - started, datasets=results,
        execution='Serial single-question official SequentialPipeline; shared generator/retriever.',
        question_manifest_sha256=manifest['question_manifest_sha256'],
        strategyqa_note=manifest['strategyqa_note']))
    return 0 if completed == 100 and all(r['status'] == 'complete' for r in results) else 1


def launch():
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid != os.getpid() and any(Path(a).name in (
            'cloud_run_smoke.py', 'cloud_multiset_batch.py', 'cloud_naive_100.py', 'main.py'
        ) for a in process.info['cmdline'] or []):
            raise RuntimeError(f'Existing experiment PID {process.pid}; refusing duplicate')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RESOURCE_REPO, text=True).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError('Unexpected official baseline source commit')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=RESOURCE_REPO, check=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'outputs' / f'{stamp}_naive_rag_fixed100'
    manifest = prepare(out)
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    with (out / 'supervisor.log').open('w') as log:
        process = subprocess.Popen([sys.executable, __file__, 'supervise', '--out', str(out)],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    record = dict(pid=process.pid, process_create_time=psutil.Process(process.pid).create_time(),
                  output_dir=str(out), source_commit=commit, method='Naive RAG', questions=100,
                  runner_sha256=sha(Path(__file__)), monitor_sha256=sha(Path(__file__).with_name('cloud_multiset_batch.py')),
                  paired_drag_inputs=str(PAIRED), manifest_sha256=manifest['question_manifest_sha256'],
                  max_runtime_seconds=7200, note='Timeout ends test, NOT Pod billing.')
    write_json(out / 'launch.json', record)
    write_json(ROOT / 'outputs/naive100-latest-launch.json', record)
    time.sleep(2)
    record['alive_after_launch'] = process.poll() is None
    print(json.dumps(record, indent=2), flush=True)
    if process.poll() is not None:
        raise RuntimeError('Supervisor exited early')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'supervise', 'worker'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode != 'launch' and args.out is None:
        parser.error('--out is required')
    if args.mode == 'launch':
        launch()
    elif args.mode == 'worker':
        sys.exit(worker(args.out))
    else:
        sys.exit(supervise(args.out, worker_script=str(Path(__file__).resolve()), run_metadata=dict(
            method='Naive RAG', version='official-baseline', source_commit=SOURCE_COMMIT,
            new_questions_requested=100)))
