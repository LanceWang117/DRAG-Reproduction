"""Run either method on saved question IDs without the old Pod's directory layout.

This submission runner is separate from the historical cloud scripts. It has
offline validation coverage; it has not been rerun on a GPU after packaging.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

from prepare_source import build_runtime

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / 'results/questions-200.jsonl'
METRICS = ['em', 'f1', 'acc', 'precision', 'recall']


def select_questions(path, dataset=None, limit=None):
    """Keep stored order and reject duplicates before importing GPU libraries."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('The question manifest is empty')
    identities = set()
    for row in rows:
        identity = (row['dataset'], row['split'], row['id'])
        if identity in identities:
            raise ValueError(f'Duplicate question: {identity}')
        identities.add(identity)
        if not row.get('question') or not row.get('golden_answers'):
            raise ValueError(f'Missing question/reference answer: {identity}')
    if dataset:
        rows = [row for row in rows if row['dataset'] == dataset]
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        rows = rows[:limit]
    if not rows:
        raise ValueError('No questions match the requested dataset')
    return rows


def resolve_paths(values, base):
    values = dict(values)
    def absolute(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (base / path).resolve())
    values['model2path'] = {key: absolute(value) for key, value in values['model2path'].items()}
    for key in ['index_path', 'corpus_path']:
        values[key] = absolute(values[key])
    return values


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(args, rows):
    overrides = resolve_paths(json.loads(args.config.read_text()), args.config.parent)
    for value in [*overrides['model2path'].values(), overrides['index_path'], overrides['corpus_path']]:
        if not Path(value).exists():
            raise FileNotFoundError(f'Resource not found: {value}; see DATA_DOWNLOADS.md')
    # Preserve the saved source defaults; use explicit overrides for this machine.
    os.environ.setdefault('OMP_NUM_THREADS', '12')
    os.environ.setdefault('MKL_NUM_THREADS', '12')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    source = build_runtime()
    sys.path.insert(0, str(source))
    started = time.monotonic()
    from flashrag.config import Config
    from flashrag.dataset import Dataset
    from flashrag.pipeline import SequentialPipeline
    from flashrag.utils import get_generator, get_retriever
    from model.drag import DebateAugmentedRAG
    from model.prompt import standard_rag_prompt_template
    from model.utils import single_agent_pred_parse

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output / stamp
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'selected-questions.json', rows)
    write_json(out / 'config-resolved.json', overrides)
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['split']), []).append(row)

    def config_for(name, split, method, count):
        settings = dict(overrides, dataset_name=name, split=[split], test_sample_num=count,
                        random_sample=False, save_dir=str(out / method),
                        save_note=f'{method}_{name}', metrics=METRICS,
                        save_intermediate_data=True, save_metric_score=True)
        return Config(str(source / 'config/base_config.yaml'), settings)

    name, split = next(iter(groups))
    config = config_for(name, split, args.method, len(groups[(name, split)]))
    print('Loading model and index...', flush=True)
    generator, retriever = get_generator(config), get_retriever(config)
    initialization = time.monotonic() - started
    print(f'Initialization: {initialization:.1f}s', flush=True)
    results = []
    methods = ['drag', 'naive'] if args.method == 'both' else [args.method]
    for method in methods:
        for (name, split), selected in groups.items():
            config = config_for(name, split, method, len(selected))
            # Golden answers go to the evaluator. The upstream pipeline builds
            # model prompts from the question and retrieved evidence, not gold.
            data = Dataset(config=config, data=[dict(id=r['id'], question=r['question'],
                           golden_answers=r['golden_answers']) for r in selected])
            if method == 'drag':
                pipeline = DebateAugmentedRAG(config, generator=generator, retriever=retriever,
                                              max_query_debate_rounds=3, max_answer_debate_rounds=3)
            else:
                pipeline = SequentialPipeline(config, standard_rag_prompt_template[name](config),
                                              generator=generator, retriever=retriever)
            elapsed_times = []
            for position, item in enumerate(data):
                single = Dataset(config=config, data=[item])
                before = time.monotonic()
                if method == 'drag':
                    pipeline.run(single, do_eval=False)
                else:
                    pipeline.run(single, do_eval=False, pred_process_fun=single_agent_pred_parse)
                seconds = time.monotonic() - before
                item.update_output('inference_elapsed_seconds', seconds)
                elapsed_times.append(seconds)
                record = dict(method=method, dataset=name, split=split, position=position,
                              elapsed_seconds=seconds, item=item.to_dict())
                # Flush each finished answer so an interrupted run retains it.
                with (out / 'checkpoints.jsonl').open('a') as log:
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                print(method, name, position + 1, '/', len(data), item.id, f'{seconds:.2f}s', flush=True)
            metrics = pipeline.evaluator.evaluate(data)
            results.append(dict(method=method, dataset=name, split=split, count=len(data),
                                metrics=metrics, inference_seconds=sum(elapsed_times),
                                mean_seconds=sum(elapsed_times) / len(data), output_dir=config['save_dir']))
            write_json(out / 'dataset-results.json', results)
    write_json(out / 'summary.json', dict(initialization_seconds=initialization,
               total_seconds=time.monotonic() - started, results=results,
               runner='submission convenience runner; not the historical experiment launcher'))
    print('Saved:', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['drag', 'naive', 'both'])
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config.example.json')
    parser.add_argument('--dataset', choices=['NQ', 'TriviaQA', 'PopQA', '2wiki', 'HotpotQA', 'StrategyQA'])
    parser.add_argument('--limit', type=int, help='First N matching questions, without resampling')
    parser.add_argument('--output', type=Path, default=PROJECT / 'new-runs')
    parser.add_argument('--validate-only', action='store_true', help='Check inputs without GPU imports or downloads')
    args = parser.parse_args()
    rows = select_questions(args.manifest, args.dataset, args.limit)
    print('Selected questions:', len(rows))
    if args.validate_only:
        print('Input validation passed; no model loaded.')
        return
    if args.method is None:
        parser.error('--method is required unless --validate-only is set')
    run(args, rows)


if __name__ == '__main__':
    main()
