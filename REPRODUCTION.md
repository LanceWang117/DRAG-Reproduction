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
