#!/usr/bin/env python3
"""Fetch the official DRAG resources without changing the official source."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/workspace/Capstone_project')
os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
REPO = ROOT / 'source_repos/Debate-Augmented-RAG'
MS = ('https://www.modelscope.cn/api/v1/datasets/'
      'hhjinjiajie/FlashRAG_Dataset/repo?Source=SDK&Revision=master&FilePath=')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fetch(url, target, expected_sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not target.exists():
        partial = target.with_name(target.name + '.part')
        if partial.with_name(partial.name + '.aria2').exists():
            raise RuntimeError(f'Wait for aria2 to complete before finalizing: {partial}')
        actual_sha = digest(partial) if partial.exists() and expected_sha else None
        if actual_sha is None or actual_sha != expected_sha:
            if expected_sha and shutil.which('aria2c'):
                command = [
                    'aria2c', '--continue=true', '--auto-file-renaming=false',
                    '--allow-overwrite=false', '--max-connection-per-server=8',
                    '--split=8', '--min-split-size=16M', '--file-allocation=none',
                    '--summary-interval=30', '--console-log-level=warn',
                    '--download-result=full', '--max-tries=5', '--retry-wait=5',
                    '--dir', str(partial.parent), '--out', partial.name, url]
            else:
                command = ['curl', '-fL', '--retry', '5', '--connect-timeout', '30',
                           '--continue-at', '-', '--output', str(partial), url]
            subprocess.run(command, check=True)
            actual_sha = digest(partial)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f'Checksum mismatch: {partial}; do not use this file')
        partial.rename(target)
    else:
        actual_sha = digest(target)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f'Existing file checksum mismatch: {target}')
    result = dict(path=str(target), url=url, bytes=target.stat().st_size,
                  sha256=actual_sha, elapsed_seconds=time.monotonic() - started)
    print(json.dumps(result), flush=True)
    return result


def public_resources():
    manifests = []
    corpus = REPO / 'wiki_corpus/wiki18_100w.jsonl'
    archive = ROOT / 'downloads/wiki18_100w_e5_index.zip'
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(fetch, url, path, sha) for url, path, sha in [
            (MS + 'retrieval_corpus%2Fwiki18_100w.jsonl', corpus,
             '43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d'),
            (MS + 'retrieval_corpus%2Fwiki18_100w_e5_index.zip', archive,
             '8cdff95467e4544a60c4224f39a32df627b8f54d6124d83fa45fc28e36899b40')]]
        manifests.extend(job.result() for job in jobs)
    index = corpus.parent / 'e5_flat_inner.index'
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist()
                   if Path(m.filename).name == 'e5_flat_inner.index']
        if len(members) != 1:
            raise RuntimeError('Expected exactly one E5 index in the archive')
        member = members[0]
        if not index.exists():
            partial = index.with_suffix('.index.part')
            with source.open(member) as inp, partial.open('wb') as out:
                shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise RuntimeError('Extracted index size mismatch')
            partial.rename(index)
        elif index.stat().st_size != member.file_size:
            raise RuntimeError('Existing index size mismatch')
    manifests.append(dict(path=str(index), bytes=index.stat().st_size,
                          extracted_from=str(archive)))
    dataset = REPO / 'dataset/NQ/dev.jsonl'
    manifests.append(fetch(
        'https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/'
        'resolve/main/nq/dev.jsonl', dataset))
    with dataset.open() as source:
        questions = [json.loads(line) for line in source if line.strip()]
    print('NQ dev questions:', len(questions), flush=True)
    print('Sample question keys:', sorted(questions[0]), flush=True)
    from huggingface_hub import model_info, snapshot_download
    revision = 'f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
    snapshot = snapshot_download(
        'intfloat/e5-base-v2', revision=revision,
        max_workers=4,
        local_dir=str(ROOT / 'models/e5-base-v2'),
        allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin'],
    )
    manifests.append(dict(model='intfloat/e5-base-v2', revision=revision,
                          path=snapshot))
    (ROOT / 'outputs/public-resource-manifest.json').write_text(
        json.dumps(manifests, indent=2) + '\n')
    print('STAGE_3_PUBLIC_RESOURCES_COMPLETE', flush=True)


def llm_resource():
    from huggingface_hub import model_info, snapshot_download
    model = 'meta-llama/Llama-3.1-8B-Instruct'
    revision = model_info(model).sha
    destination = snapshot_download(
        model, revision=revision,
        max_workers=4,
        local_dir=str(ROOT / 'models/Llama-3.1-8B-Instruct'),
        allow_patterns=['*.json', '*.safetensors', 'tokenizer.model',
                        'LICENSE', 'USE_POLICY.md', 'README.md'],
    )
    result = dict(model=model, revision=revision, path=destination)
    (ROOT / 'outputs/llm-resource-manifest.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('STAGE_3_LLM_RESOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('resource', choices=['public', 'llm'])
    args = parser.parse_args()
    if args.resource == 'public':
        public_resources()
    else:
        llm_resource()
