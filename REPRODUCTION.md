# Running and checking the reproduction

## Inspect existing results first

No GPU or third-party Python packages are needed for these commands:

```bash
python3 scripts/verify_evidence.py
python3 -m unittest discover -s scripts -p test_submission.py -v
python3 scripts/run_evaluation.py --validate-only
```

The first command checks the preserved evidence hashes, question identities,
answers, per-question EM/F1, retrieved passage records, and aggregate scores.
It does not regenerate answers or replace the saved experiment results.

## Linux environment

The GPU path requires Linux and an NVIDIA CUDA GPU. Use an isolated Python
environment; do not replace the Pod template's system packages.

```bash
conda create -n drag python=3.11 -y
conda activate drag
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements.txt` is the cleaned version list used for cloud setup. The full
observed environment is retained in `runtime-pip-freeze.txt` and
`runtime-conda-explicit.txt`; it can contain build-specific paths and should not
be treated as a portable pip requirements file. Driver compatibility and
available wheels still need checking on a new machine.

Download the models, corpus, and index as described in `DATA_DOWNLOADS.md`, then:

```bash
cp config.example.json config.local.json
```

Edit model/corpus/index paths in the local copy. The portable runner resolves
relative paths against that configuration file's directory. The local copy is
ignored by Git. Never add a Hugging Face token to it.

## One-question smoke test

Start with one question before launching a batch. These commands load the full
model and index and therefore take several minutes even for one question.

```bash
python scripts/run_evaluation.py --method naive --dataset NQ --limit 1 \
  --config config.local.json
python scripts/run_evaluation.py --method drag --dataset NQ --limit 1 \
  --config config.local.json
```

Do not run them concurrently on the same GPU. Each command writes a new folder
under `new-runs/`. For the same selected question and one shared initialization:

```bash
python scripts/run_evaluation.py --method both --dataset NQ --limit 1 \
  --config config.local.json
```

The portable runner was added when preparing this repository. Input validation
and evidence tests pass locally, but we have not rerun this new wrapper on a GPU.
The uploaded results came from the historical scripts listed below, not from
this new wrapper. The DRAG implementation and prompt templates are the saved
source snapshot, not rewritten algorithms.

## The fixed 200-question sample

```bash
python scripts/run_evaluation.py --method both --config config.local.json
```

This explicitly selects a substantial GPU job. It preserves the saved manifest
order and does not resample. With both methods it shares model/index instances,
but uses separate pipelines and result folders. DRAG uses three retrieval
rounds and three response rounds as upper limits.

The runner times each serial inference call, flushes a checkpoint after each
answer, and evaluates each completed dataset using FlashRAG. Reference answers
are for scoring; they are not inserted into the model's question/evidence prompt.
All five saved metrics are deterministic string-based calculations, not extra
LLM evaluation calls. The published discussion focuses on EM and F1.

Output files include the selected inputs, resolved configuration, checkpoints,
FlashRAG intermediate data and metrics, and an overall summary. The portable
runner does not have the historical supervisor's resource sampling or timeout.
Keep the terminal open, or use an appropriate session manager if disconnecting.

## How the reported runs were made

| Script | Purpose |
| --- | --- |
| `cloud_run_smoke.py` | Invokes upstream `main.py`, captures output, samples CPU/RAM/GPU, and enforces a timeout; used for NQ 50 |
| `cloud_multiset_batch.py` | Shared initialization and five datasets of 10 DRAG questions; also supplies the background supervisor |
| `cloud_naive_100.py` | Serial official Naive pipeline on the exact first 100 DRAG inputs |
| `cloud_expand_200.py` | Adds PopQA 30, 2Wiki 30, HotpotQA 25, StrategyQA 15, excluding existing IDs; runs both methods |
| `cloud_fetch_resources.py`, `cloud_fetch_benchmarks.py` | Original downloads and provenance recording |

These scripts are preserved with their historical absolute paths and output
directory references so the record can be audited. They expect the old
`/workspace/Capstone_project` tree, with original and fixed source directories.
Use the portable runner for a new checkout rather than changing those historical
files. `test_cloud_*` files belong to those scripts.

`src/main.py` is the authors' experiment entry point, not our code. It selects
the requested method and uses FlashRAG to load questions and evaluate. It reads
the `dev` split directly. Our multi-dataset wrappers call the library classes
instead so PopQA/test and StrategyQA/train retain their split names.

## Judge fix tests

```bash
cd src
python -m unittest discover -s tests -v
```

Parser tests use the standard library. Integration tests use fake model and
retriever objects; they require the runtime imports but do not load weights.
Without those imports some tests skip. The original cloud validation output is
saved as `results/judge-validation-tests-v1.0.1.log`.

## Version and timing notes

The reported fixed source snapshot is `3870c0e...`, based on upstream
`c035ee1...`. See `judge-fix.patch` for the complete source change and
`JUDGE_FIX.md` for its limits. Both comparison methods use the same model,
corpus, index, generation settings, and selected questions.

Inference times exclude initialization and are taken from serial runs. The NQ
DRAG batch time comes from its progress log, not exact per-question timing.
Resource sampling is coarse and process-tree RSS can double-count shared pages.
The newly selected extra 100 questions took about 56m47s for both methods
together, including shared initialization and supervisor overhead. This is not
the total time spent on all earlier setup, pilots, and reruns.
