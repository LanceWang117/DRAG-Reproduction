"""Targeted paired extension: 100 new questions, both methods, shared initialization."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from cloud_multiset_batch import ROOT, RESOURCE_REPO, SOURCE_REPO, SOURCE_COMMIT
from cloud_multiset_batch import TimedDataset, sha, supervise, write_json

PREVIOUS = ROOT / 'outputs/20261017T015401Z_drag_extra50_v1.0.1'
ALLOCATION = [('PopQA', 'test', 30), ('2wiki', 'dev', 30),
              ('HotpotQA', 'dev', 25), ('StrategyQA', 'train', 15)]


def prepare(out):
    prior = [json.loads(s) for s in (PREVIOUS / 'question-manifest-100.jsonl').read_text().splitlines()]
    assert len(prior) == 100
    excluded = {(r['dataset'], r['split'], r['id']) for r in prior}
    out.mkdir(parents=True, exist_ok=False)
    entries, added = [], []
    for i, (name, split, count) in enumerate(ALLOCATION):
        source = RESOURCE_REPO / f'dataset/{name}/{split}.jsonl'
        rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        eligible = [r for r in rows if (name, split, r['id']) not in excluded]
        chosen = random.Random(20261017 + i).sample(eligible, count)
        assert len({r['id'] for r in chosen}) == count
        assert all(r.get('question') and r.get('golden_answers') for r in chosen)
        path = out / f'fixed-datasets/{name}/{split}.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(''.join(json.dumps(r) + '\n' for r in chosen))
        entries.append(dict(dataset=name, split=split, count=count,
                            selection_seed=20261017 + i, source_sha256=sha(source),
                            subset_sha256=sha(path), ordered_ids=[r['id'] for r in chosen]))
        added.extend(dict(dataset=name, split=split, position=j, id=r['id'],
                          question=r['question'], golden_answers=r['golden_answers'])
                     for j, r in enumerate(chosen))
    assert len(added) == 100
    assert not excluded.intersection((r['dataset'], r['split'], r['id']) for r in added)
    for filename, rows in [('question-manifest-new100.jsonl', added),
                           ('question-manifest-200.jsonl', prior + added)]:
        (out / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_json(out / 'sample-manifest.json', dict(
        entries=entries, previous_manifest=str(PREVIOUS / 'question-manifest-100.jsonl'),
        previous_manifest_sha256=sha(PREVIOUS / 'question-manifest-100.jsonl'),
        new_question_count=100, combined_count=200,
        rationale='Targeted expansion based on pilot discrepancies/DRAG gains; not balanced or unbiased confirmation.',
        strategyqa_note='Train split used for exploratory evaluation only; no training.',
        source_commit=SOURCE_COMMIT))


def worker(out):
    os.environ.update(HF_HOME=str(ROOT / 'cache/huggingface'), HF_HUB_ENABLE_HF_TRANSFER='0',
                      TMPDIR=str(ROOT / 'cache/tmp'), OMP_NUM_THREADS='12', MKL_NUM_THREADS='12',
                      TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(SOURCE_REPO))
    from flashrag.config import Config
    from flashrag.dataset import Dataset
    from flashrag.utils import get_generator, get_retriever
    from flashrag.pipeline import SequentialPipeline
    from model.drag import DebateAugmentedRAG
    from model.prompt import standard_rag_prompt_template
    from model.utils import single_agent_pred_parse
    manifest = json.loads((out / 'sample-manifest.json').read_text())
    started = time.monotonic()

    def config_for(entry, method):
        return Config(str(SOURCE_REPO / 'config/base_config.yaml'), {
            'model2path': {'e5': str(ROOT / 'models/e5-base-v2'),
                           'llama3-8B-instruct': str(ROOT / 'models/Llama-3.1-8B-Instruct')},
            'framework': 'vllm', 'generator_model': 'llama3-8B-instruct', 'gpu_id': '0',
            'faiss_gpu': False, 'seed': 2024,
            'index_path': str(RESOURCE_REPO / 'wiki_corpus/e5_flat_inner.index'),
            'corpus_path': str(RESOURCE_REPO / 'wiki_corpus/wiki18_100w.jsonl'),
            'dataset_name': entry['dataset'], 'split': [entry['split']],
            'data_dir': str(out / 'fixed-datasets'), 'random_sample': False,
            'test_sample_num': entry['count'], 'save_dir': str(out / method / 'official-results'),
            'save_note': method + '_targeted_extension', 'save_intermediate_data': True,
            'save_metric_score': True, 'metrics': ['em', 'f1', 'acc', 'precision', 'recall']})

    config = config_for(manifest['entries'][0], 'DRAG')
    print('SHARED_INIT_START', flush=True)
    generator, retriever = get_generator(config), get_retriever(config)
    initialization = time.monotonic() - started
    print('SHARED_INIT_COMPLETE', initialization, flush=True)
    results = []
    for method in ['DRAG', 'Naive']:
        target = out / method
        target.mkdir(exist_ok=True)
        for entry in manifest['entries']:
            name, split = entry['dataset'], entry['split']
            config = config_for(entry, method)
            path = out / f'fixed-datasets/{name}/{split}.jsonl'
            assert sha(path) == entry['subset_sha256']
            dataset = Dataset(config=config, dataset_path=str(path))
            tracked = TimedDataset(dataset, target, name, split)
            if method == 'DRAG':
                pipeline = DebateAugmentedRAG(config, generator=generator, retriever=retriever,
                                              max_query_debate_rounds=3, max_answer_debate_rounds=3)
                pipeline.run(tracked)
            else:
                pipeline = SequentialPipeline(config, standard_rag_prompt_template[name](config),
                                              generator=generator, retriever=retriever)
                for item in tracked:
                    pipeline.run(Dataset(config=config, data=[item]), do_eval=False,
                                 pred_process_fun=single_agent_pred_parse)
                pipeline.evaluator.evaluate(dataset)
            metrics = {s.partition(':')[0]: float(s.partition(':')[2])
                       for s in (Path(config['save_dir']) / 'metric_score.txt').read_text().splitlines()
                       if ':' in s}
            assert len(tracked.elapsed) == entry['count'] and len(metrics) == 5
            result = dict(method=method, dataset=name, split=split, questions=len(tracked.elapsed),
                          metrics=metrics, inference_seconds=sum(tracked.elapsed),
                          mean_seconds=sum(tracked.elapsed) / len(tracked.elapsed),
                          output_dir=config['save_dir'])
            results.append(result)
            write_json(out / 'dataset-results.json', results)
            print('DATASET_COMPLETE', json.dumps(result), flush=True)
    write_json(out / 'experiment-summary.json', dict(status='complete', results=results,
               initialization_seconds=initialization, worker_seconds=time.monotonic() - started,
               new_questions_per_method=100, combined_questions_per_method=200))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'worker', 'supervise'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode == 'launch':
        import psutil
        for p in psutil.process_iter(['pid', 'cmdline']):
            if p.pid != os.getpid() and any(Path(a).name in ('cloud_expand_200.py', 'cloud_naive_100.py',
                   'cloud_multiset_batch.py', 'cloud_run_smoke.py') for a in p.info['cmdline'] or []):
                raise RuntimeError(f'Existing experiment: {p.pid}')
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE_REPO, text=True).strip() == SOURCE_COMMIT
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=SOURCE_REPO, check=True)
        out = ROOT / 'outputs' / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_paired_extension100')
        prepare(out)
        with (out / 'supervisor.log').open('w') as log:
            p = subprocess.Popen([sys.executable, __file__, 'supervise', '--out', str(out)],
                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        record = dict(pid=p.pid, output_dir=str(out), runner_sha256=sha(Path(__file__)),
                      questions_per_method=100, max_runtime_seconds=7200)
        write_json(out / 'launch.json', record)
        write_json(ROOT / 'outputs/extension200-latest-launch.json', record)
        print(json.dumps(record), flush=True)
    elif args.mode == 'worker':
        sys.exit(worker(args.out))
    else:
        sys.exit(supervise(args.out, worker_script=str(Path(__file__).resolve()),
                           run_metadata=dict(method='DRAG_and_Naive', new_questions_requested=100)))
