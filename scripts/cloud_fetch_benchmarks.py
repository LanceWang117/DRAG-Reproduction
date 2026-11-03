"""Fetch official FlashRAG evaluation splits, preserving original split names."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path('/workspace/Capstone_project')
REPO = ROOT / 'source_repos/Debate-Augmented-RAG'
HUB = 'https://huggingface.co'
DATASET = 'RUC-NLPIR/FlashRAG_datasets'
# NQ is downloaded by cloud_fetch_resources.py; do not start a competing writer.
NAMES = {'triviaqa': 'TriviaQA', 'hotpotqa': 'HotpotQA',
         '2wikimultihopqa': '2wiki', 'popqa': 'PopQA', 'strategyqa': 'StrategyQA'}


def read_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


info = read_json(f'{HUB}/api/datasets/{DATASET}')
revision = info['sha']
files = [item['rfilename'] for item in info['siblings']]
manifest = {'dataset_repository': DATASET, 'revision': revision, 'splits': []}
for folder, local in NAMES.items():
    selected = [p for p in files if p.split('/')[0].lower() == folder
                and p.split('/')[-1] in ('dev.jsonl', 'test.jsonl')]
    if not selected and folder == 'strategyqa':
        selected = [p for p in files if p == 'strategyqa/train.jsonl']
    if not selected:
        raise RuntimeError(f'No evaluation split found for {folder}')
    for remote in sorted(selected):
        destination = REPO / 'dataset' / local / Path(remote).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f'{HUB}/datasets/{DATASET}/resolve/{revision}/{remote}'
        with urllib.request.urlopen(url, timeout=90) as response:
            contents = response.read()
        rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
        assert rows and all('question' in r and 'golden_answers' in r for r in rows)
        temporary = destination.with_suffix('.jsonl.part')
        temporary.write_bytes(contents)
        temporary.replace(destination)
        record = {'dataset': local, 'upstream': remote, 'split': destination.stem,
                  'questions': len(rows), 'bytes': len(contents), 'url': url,
                  'sha256': hashlib.sha256(contents).hexdigest(),
                  'path': str(destination)}
        if folder == 'strategyqa' and destination.stem == 'train':
            record['warning'] = ('Upstream provides train only. Preserved as train; '
                                 'evaluation selection/protocol still needs review.')
        manifest['splits'].append(record)
        print(json.dumps(record), flush=True)
(ROOT / 'outputs/benchmark-resource-manifest.json').write_text(
    json.dumps(manifest, indent=2) + '\n')
print('OTHER_FIVE_BENCHMARKS_DOWNLOADED; NQ is handled by public resource script.', flush=True)
