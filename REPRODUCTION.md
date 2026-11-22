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

Budget flags (J08): `response_rounds` (0-3, null inherits `max_answer_debate_rounds`)
sets the answer-stage round count; `agent_max_tokens` and `moderator_max_tokens` are
passed as per-call `max_tokens` (null leaves the generator/config default);
`concise_agents` asks each answer agent for at most two sentences and keeps the answer
marker. With a budget, `finish_reason` is inferred as `length` when the output reaches
the cap and `stop` when it stays under, and `truncated` marks a budgeted answer that lost
"The answer is:".

`--resume` skips finished questions, drops a torn last checkpoint line, and refuses a
different variant or `--interleave` setting. The logs of a resumed run equal those of an
uninterrupted one. The fake backend's answers are meaningless: it exists to test the path.

### Local smoke run

`scripts/smoke_local.py` runs 1 to 5 dev questions end to end with a **real** small model,
to catch prompt, parsing and stage-flow bugs on real model text before the A100 stage.
It is never a source of results: the default model, `llama3.2:3b`, is a different family
and size than the A100's Llama-3.1-8B BF16, so nothing it prints is comparable, and no
accuracy is reported. Its output goes to `new-runs/local-smoke/` (ignored by git).

```bash
python scripts/smoke_local.py --questions 3        # improvements/configs/local-smoke.json
python scripts/smoke_local.py --questions 3 \
    --variant improvements/configs/baseline.json improvements/configs/cached.json
python scripts/run_variant.py --variant improvements/configs/baseline.json \
    --backend local --limit 3 --profile            # with an Ollama server already running
```

- **Generator:** Ollama's OpenAI-compatible `/v1/chat/completions`, standard library only.
  It sends message lists (not chat strings) at temperature 0, `max_tokens` 256 and seed
  2024, retries a timeout, dropped connection or HTTP 5xx at most twice, and returns
  outputs in input order. The OpenAI endpoint cannot set the context size, so every
  prompt is fitted to 4096 tokens (a conservative 3 characters per token) by cutting the
  middle of its longest message; the summary counts trimmed prompts.
- **Retriever:** replays `results/retrieval-200.jsonl`. A stored query returns its stored
  passages (the question text maps to the Naive list when the DRAG map lacks it); a new
  query ranks every stored passage by token overlap. No FAISS, no corpus.
- **Preflight:** refuses to start with under 3 GiB of free VRAM, 4 GiB of free RAM or
  1 GiB of free disk. If no server answers it runs `ollama serve` with one parallel
  request, one loaded model and a 4096-token context, and stops it (after unloading the
  model) when it finishes, even on failure. A server that was already running is never
  stopped. A missing model is pulled only if its tag is 4B parameters or fewer and 10 GiB
  of disk are free; `--no-pull` turns that off.
- **Summary:** per variant, `PASS` or `FAIL` with completed answers, parse failures (a final
  answer without `The answer is:`), empty model outputs and log-schema errors. A variant
  fails when the run did not complete, a log row misses a field, a model call came back
  empty, or no answer at all carried the marker. Exit status: 0 all passed, 1 a variant
  failed, 2 the preflight or the server refused.

The generator does not request log-probs. Ollama 0.34.4 returns them on the same endpoint
(`logprobs: true`, OpenAI-shaped, checked against a live server), so a log-prob Judge can be
smoke-tested locally once it asks for them. Smoke runs prove nothing about FlashRAG, vLLM,
FAISS, the Llama-3.1 tokenizer or GPU memory; those stay on the A100.

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

### Anchored evidence

`evidence_store="anchored"` keeps an append-only, `doc_id`-keyed evidence store per
question. The original question's passages are anchored, so a query Optimization or
Expansion can add passages and supersede later queries but cannot evict them. Duplicate
passages are dropped and each surviving passage gets a global label `E{qid}-{rank}`
(query ordinal, first-seen order). `evidence_delimiters=true` renders every passage as a
`<document id="E..." title="...">` block with a data-not-instructions notice;
`rewrite_novelty_audit=true` logs entities and numbers a rewrite invents. All three
default to the baseline (`dict`, `false`, `false`).

`scripts/replay_evidence.py` estimates the recall effect from the committed passages.
It substitutes the Naive arm's passages for the original retrieval an Optimization may
have deleted, so its output is an approximation, not a rerun. On the 175 non-StrategyQA
dev questions it gives 92/175 for the stored final pool and 102/175 for the anchored
union (gold-string recall, 0.5257 vs 0.5829).

```bash
python scripts/replay_evidence.py
```

### Structured Opponent actions and Judge visibility

`opponent_action_schema=true` replaces the free-text Opponent instruction with a
reasoning-plus-JSON schema: `{action, source_query_id, new_query, reason}`, where
`action` is `optimize`, `expand` or `no_op`. The parser takes the last balanced JSON
object (fences and trailing text are tolerated), rejects a second action object, unknown
or missing fields, wrong types and an empty `new_query`; the validator requires 3-30
tokens, no duplicate of an active query, an existing source ID for `optimize`, and a
reason when a source entity is dropped. Queries carry stable IDs (`Q1`, `Q2`, ...) in
first-seen order, so `optimize` targets the named ID instead of the baseline's
Levenshtein "closest key". The candidate pool is built and validated before one commit;
any failure leaves the pool unchanged and logs `{raw, parsed, error}` as an
`opponent_action` decision. An invalid action counts as `no_op` for control flow.

`judge_sees_refined_query=true` changes only the query-stage Judge's `Question:` line,
from the original question to the latest active refined query. Both flags default to
`false`, which keeps the v1.0.1 transcript byte for byte.

```json
{"opponent_action_schema": true, "judge_sees_refined_query": true}
```

### Token budget and prompt layout

Three flags control context growth and prompt reuse; all default to the baseline:

- `context_token_budget` (integer or `null`): selects evidence MMR-style by relevance
  per token (retrieval rank is the relevance signal), subtracts a redundancy penalty
  against already selected passages, and keeps at most two chunks per article. The
  budget covers the header, question, evidence and history; role instructions are not
  part of it. Anchored passages are never dropped: if they alone exceed the budget, the
  lowest-rank anchored passage is truncated and a `context_budget` decision row records
  the warning. `null` keeps every passage.
- `prompt_layout` (`role_first` | `evidence_first`): `evidence_first` starts every
  role's system message with one byte-identical `header + question + evidence` prefix,
  then appends the unchanged role instruction. Evidence is append-only under
  `evidence_store="anchored"`, so round r's prefix is a prefix of round r+1's.
- `enable_prefix_caching` (boolean): passed to vLLM as `enable_prefix_caching=True`
  when the FlashRAG backend builds its generator. It stays off by default because a
  cache-hit prefill is not bit-identical to a cold prefill; only E-job variants that
  pass the A/A floor should turn it on.

```json
{"context_token_budget": 2500, "prompt_layout": "evidence_first",
 "enable_prefix_caching": true}
```

### GPU retrieval

Three flags control retrieval acceleration; all defaults reproduce the baseline:

| Flag | Default | Meaning |
| --- | --- | --- |
| `faiss_gpu` | `false` | clone the FAISS index onto the visible GPUs in fp16 |
| `gpu_memory_utilization` | `0.85` | vLLM's per-GPU reservation; lower it when the index is on the GPU |
| `omp_threads` | `12` | `OMP_NUM_THREADS` and `MKL_NUM_THREADS` for the CPU parts |

`OMP_NUM_THREADS` and `MKL_NUM_THREADS` are set from `omp_threads` at the start of
backend construction, before `Config` (which imports torch), FAISS or vLLM are imported.
FAISS reads the OpenMP variable for its CPU threads and MKL reads its setting when its
shared library loads, so setting them later can be ignored. This also applies when
`faiss_gpu` is false; a thread-only variant needs no GPU.

With `faiss_gpu=true` the runner builds the retriever before the generator.
`DenseRetriever` clones the index onto the GPUs in fp16 (`useFloat16=True`,
`shard=True`) inside its constructor, while FlashRAG's vLLM generator reserves its KV
pool inside its constructor. Generator-first would reserve the pool before the index
copy and can exhaust the card. The CPU path keeps the recorded generator-first order. A
CPU-only FAISS build, or a card that FAISS cannot see, fails immediately with the
install pointer; the runner never falls back to CPU silently.

`scripts/preflight_gpu.py` refuses a run before anything is loaded when the memory
cannot fit:

```
required = gpu_memory_utilization * GPU total
         + index_bytes / 2          # FAISS keeps an fp16 copy
         + 0.5 GiB                  # E5 query encoder
         + 3 GiB                    # allocator/CUDA context margin
```

An 80 GiB A100 with a 64.5 GiB fp32 index and `gpu_memory_utilization=0.45` needs
71.75 GiB and passes; `0.85` needs 103.75 GiB and fails with the shortfall and the
suggested fix. The script reads `nvidia-smi --query-gpu=memory.total` (summing visible
cards); `--gpu-total-gib` is the offline path used by the PC tests.

#### Install FAISS GPU on the Pod

The pinned environment is Python 3.11, `torch==2.5.1` (CUDA 12.4 wheels) and
`numpy==1.26.4`. Install the conda-forge CUDA 12 build. Do not use PyPI
`faiss-gpu-cu12`: its cp311 builds start at 1.10.0 and require `numpy>=2`, which conflicts
with the pinned numpy. FlashRAG's README suggests
`conda install -c pytorch -c nvidia faiss-gpu=1.8.0`, but that channel's builds predate
Python 3.11, so the pinned conda-forge build below is the reproducible route. Each
command is **[E]** until it has run on the A100.

```bash
conda activate <drag-env>                                            # [E]
pip uninstall -y faiss-cpu                                           # [E] conflicts with faiss-gpu
conda install -y -c conda-forge "faiss=1.8.0=py311cuda120hebc9808_1_cuda"  # [E]
python -c "import faiss; print(faiss.__version__, faiss.get_num_gpus())"   # [E] expect: 1.8.0 1
```

The conda-forge package constrains `faiss-cpu <0.0a0`, hence the explicit pip removal,
and its `numpy >=1.23.5,<2` dependency matches `requirements.txt`. A different Python
version needs a different build string. Then check the planned memory and run one
question:

```bash
INDEX=wiki_corpus/e5_flat_inner.index
python scripts/preflight_gpu.py --index-bytes "$(stat -c%s "$INDEX")" \
    --gpu-memory-utilization 0.45                                    # [E]
python scripts/run_variant.py --variant improvements/configs/s2-gpu-faiss.json \
    --config config.local.json --dataset NQ --limit 1 --profile       # [E]
```

`improvements/configs/s2-gpu-faiss.json` sets `faiss_gpu=true`,
`gpu_memory_utilization=0.45` and `omp_threads=12`. E01 measures the speed and memory;
this section only makes the configuration safe and reproducible.
