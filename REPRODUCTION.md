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
`(index, encoder, top_k, normalised query)`, with its documents and scores. A miss sends
one `batch_search` for all missing queries in the request; repeats within a request are
asked once, the first caller spelling is what the retriever sees, and the result order is
preserved. Documents and score lists are deep-copied on the way out. On load a torn final
line is truncated and any other unreadable line is skipped with a warning, so a partial
write cannot poison later records. A relative path resolves against the run directory,
and a missing parent directory is created. Leave the flag at `null` (the default) for the
bare baseline call: no proxy, no file.
