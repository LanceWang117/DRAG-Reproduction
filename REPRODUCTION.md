# Reproduction commands

## Check saved results without a GPU

```bash
python3 scripts/verify_evidence.py
python3 -m unittest discover -s scripts -p test_submission.py -v
python3 scripts/run_evaluation.py --validate-only
```

The checker validates the compact evidence hashes, fixed inputs, passage IDs
and text, and aggregates the stored per-question scores. It does not regenerate
answers or use an LLM for scoring.

## Linux setup

The tested platform was an NVIDIA A100 on Linux. Use an isolated environment:

```bash
conda create -n drag python=3.11 -y
conda activate drag
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp config.example.json config.local.json
```

Follow DATA_DOWNLOADS.md for the model, corpus, and index. Edit their paths in
`config.local.json`; relative paths resolve from the config file's directory.
The tested versions and full installed package lists are in ENVIRONMENT.md.

## One question, then a batch

```bash
python scripts/run_evaluation.py --method both --dataset NQ --limit 1 --config config.local.json
```

`--method` accepts `drag`, `naive`, or `both`. `--limit` selects the first
N matching questions from the saved list, without resampling. To run all 200:

```bash
python scripts/run_evaluation.py --method both --config config.local.json
```

This starts a substantial GPU job. Do not run two jobs on the same GPU.
Initialization takes several minutes; later questions reuse the model/index.
New output goes under ignored `new-runs/`, leaving recorded results unchanged.

The runner prepares ignored `.runtime/drag/` by copying the original source
and overlaying our two Judge files. The original source directory is not edited.
Each finished answer is flushed to a checkpoint. FlashRAG calculates EM/F1
and other saved metrics after each dataset; gold answers are not model inputs.

This portable runner was added for the repository. It has offline checks but
has not been rerun on a GPU. The reported experiments used the historical cloud
wrappers, preserved with the full local evidence. Those wrappers added timing,
resource sampling, checkpointing, and paired selection around the original
DRAG and FlashRAG classes. The portable runner has no automatic timeout or
resource monitor; keep the session open or use a session manager.

## Judge tests

```bash
python3 -m unittest discover -s src/reproduction_changes/tests -v
```

Parser tests need only Python. Integration tests use fake model/retriever calls,
but still need the Linux runtime imports; they skip when those are unavailable.
No test loads the real Llama weights or FAISS index.

The authors' own entry point remains `src/original_drag/main.py`. It directly
reads the dev split. Our runner calls the same classes to support PopQA/test
and StrategyQA/train without relabeling those splits.

## Variant runner

`scripts/run_variant.py` runs a variant of the pipeline (the `drag_plus` package and a
Naive arm) and records per-call timings. It is separate from `scripts/run_evaluation.py`,
which stays the reference for the recorded baseline numbers.

A variant file names the flags to set and the arms to run
(`improvements/configs/baseline.json` is the baseline with every flag at its default):

```bash
python scripts/run_variant.py --variant improvements/configs/baseline.json --validate-only
python scripts/run_variant.py --variant improvements/configs/baseline.json \
    --backend fake --limit 5 --profile --output /tmp/variant-demo     # offline, no GPU
python scripts/run_variant.py --variant improvements/configs/baseline.json \
    --config config.local.json --dataset NQ --limit 5 --profile      # needs the A100 stack
python scripts/run_variant.py --variant improvements/configs/baseline.json \
    --config config.local.json --resume new-runs/<run-directory>     # after an interruption
```

Each run writes `new-runs/<UTC>-<variant>/` (ignored by git) with:
- `checkpoints.jsonl`: one fsynced line per finished (arm, question); the durable record;
- `transcripts.jsonl`: every prompt and raw output in order with its sha256, and `pred`
  under both extractors;
- `stage-timings.jsonl`: one row per generator or retriever call, only with `--profile`;
  `t_start` is seconds since the question's first call;
- `run-manifest.json`: variant, flags, commit, package versions, thread settings, backend,
  start and end, exit status and every resume;
- `summary.json`: the metrics for each arm and dataset.

`--resume` skips finished questions, drops a torn last checkpoint line, and refuses a
different variant or `--interleave` setting. The logs of a resumed run equal those of an
uninterrupted one. The fake backend's answers are meaningless: it exists to test the path.

### Retrieval cache

Setting `retrieval_cache` to a path wraps the backend's retriever in one caching proxy
that **both arms share** (`improvements/configs/cached.json` is a ready variant):

```bash
python scripts/run_variant.py --variant improvements/configs/cached.json \
    --backend fake --limit 5 --output /tmp/variant-cached
```

The proxy exists because FlashRAG's `search` and `batch_search` are not equivalent at the
pinned version: `DenseRetriever._search` passes a bare string to `Encoder.encode`, which
slices it by character and so embeds only the first 64 characters
(`flashrag/retriever/retriever.py:389-402`), while `_batch_search` wraps the string in a
list and embeds all of it (`:404-428`). That is the cause of B2: 32 of the 124 unchanged
pools differed, and every one of them has a query longer than 57 characters. The proxy's
`search` is defined as `batch_search([query])[0]`, so the two call shapes cannot diverge.

Each query is stored in `<path>` as one JSONL line keyed by
`(index, encoder, top_k, normalised query, config)`, with its documents and scores.
`config` is a canonical fingerprint of the retriever settings (`retriever`, the k values,
both weights and floors, the BM25 index, `rerank_k` and its model, `per_title_cap`), so a
dense entry can never answer a hybrid query that happens to share the other four fields.
A miss sends one `batch_search` for all missing queries in the request; repeats within a
request are asked once, the first caller spelling is what the retriever sees, and the
result order is preserved. Documents and score lists are deep-copied on the way out. On
load a torn final line is truncated and any other unreadable line is skipped with a
warning, so a partial write cannot poison later records. A relative path resolves against
the run directory, and a missing parent directory is created. Leave the flag at `null`
(the default) for the bare baseline call: no proxy, no file.

### Hybrid retrieval

`retriever = "hybrid"` fuses E5's dense top-K with BM25's top-K by weighted reciprocal
rank fusion (Cormack et al., SIGIR 2009: `sum_r w_r / (k + rank_r(d))`, k = 60), then
optionally reranks the head with a cross-encoder and caps passages per article:

| Flag | Default | Meaning |
| --- | --- | --- |
| `retriever` | `e5` | `hybrid` composes the retriever below; `e5` is the bare dense one |
| `dense_top_k`, `bm25_top_k` | `20`, `20` | candidates each source contributes to the fusion |
| `hybrid_top_k` | `3` | passages returned per query (the baseline depth) |
| `rrf_k`, `dense_weight`, `bm25_weight` | `60`, `1.0`, `1.0` | the fusion itself; weight 0 drops a source |
| `dense_score_floor`, `bm25_score_floor` | `null` | drop a source's candidates below this score before fusion |
| `bm25_index` | `null` | directory saved by `BM25.save(..., corpus=[{"id", "contents"}])` |
| `rerank_k` | `null` | rerank this many fused candidates with a cross-encoder; `null` imports none |
| `rerank_model` | `BAAI/bge-reranker-base` | cross-encoder id, MIT, `max_length=512` |
| `per_title_cap` | `null` | at most N passages per article, the title being the first line of `contents` |

Ties break on the passage id, a source that returns fewer than k results contributes the
ranks it has, and each query returns at most `hybrid_top_k` passages. The BM25 adapter
loads the index memory-mapped on its first query, so an `e5` run never imports `bm25s`
and a hybrid run without `rerank_k` never imports `sentence-transformers`. E04 builds the
real index; nothing here assumes its size.

`scripts/eval_retrieval.py` scores candidate lists alone, so it can score a fresh run or
the committed baseline passages:

```bash
python scripts/eval_retrieval.py --retrieval results/retrieval-200.jsonl
python scripts/eval_retrieval.py --retrieval results/retrieval-200.jsonl \
    --timings new-runs/<run>/stage-timings.jsonl --out /tmp/retrieval.json
```

It reports Containment@k (gold normalised by J01's scorer, measured on the question plus
the top-k passages), MRR, nDCG@k, the unique-title ratio, and p50/p99 retrieval latency
from the timing log. On the committed passages it gives 114/200 for each arm and 115/200
for their union, which is how finding B17 recorded containment. Passages alone give
105/200, so the two numbers are not interchangeable.
