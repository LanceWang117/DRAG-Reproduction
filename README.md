# DRAG reproduction

Capstone reproduction of [Debate-Augmented RAG (ACL 2025)](https://aclanthology.org/2025.acl-long.770/).
We ran DRAG and Naive RAG on the same 200 questions using Llama 3.1 8B,
the full Wikipedia corpus, and the E5 FAISS index.

## Where to look

| Location | Content / ownership |
| --- | --- |
| `src/original_drag/` | Authors' original repository, unchanged |
| `src/reproduction_changes/` | Our Judge fix and commented tests |
| `scripts/` | Our runner, source preparation, and offline checks |
| `results/` | Fixed questions, both answers, retrieval records, and scores |
| `improvements/` | Reserved for later work; no improvement claimed yet |

Original repository: [Huenao/Debate-Augmented-RAG](https://github.com/Huenao/Debate-Augmented-RAG),
commit `c035ee1b7e6bc9fa67d5a104df1c2175a8300666`.
The source changes use the Judge validation policy tested as local v1.0.1.
The original license is retained; see [src/README.md](src/README.md) for the
exact distinction between original and added code.

## Results

| Dataset | Questions | Naive EM / F1 | DRAG EM / F1 |
| --- | ---: | --- | --- |
| NQ | 50 | 44.00 / 54.72 | 34.00 / 50.56 |
| TriviaQA | 10 | 60.00 / 67.25 | 70.00 / 72.86 |
| PopQA | 40 | 45.00 / 49.47 | 55.00 / 59.76 |
| 2Wiki | 40 | 15.00 / 18.00 | 35.00 / 38.13 |
| HotpotQA | 35 | 28.57 / 36.67 | 34.29 / 46.84 |
| StrategyQA | 25 | 64.00 / 64.00 | 64.00 / 64.00 |

Scores are percentages. Mean serial inference time, excluding initialization:
Naive 4.86s/question; DRAG 28.23s/question. The gains on multi-hop questions
partly follow the paper's trends, but this is not a full paper-scale reproduction.

The sample is small and uneven. The extra 100 questions were selected after
the first pilot, and StrategyQA uses its available train split. Our DRAG results
include the [Judge parsing fix](JUDGE_FIX.md), not unchanged upstream code.
No statistically established improvement or new routing method is claimed.

## Check or run

```bash
python3 scripts/verify_evidence.py
python3 -m unittest discover -s scripts -p test_submission.py -v
python3 scripts/run_evaluation.py --validate-only
```

These commands need no GPU. See [REPRODUCTION.md](REPRODUCTION.md) to run a model,
[ENVIRONMENT.md](ENVIRONMENT.md) for the combined environment record, and
[DATA_DOWNLOADS.md](DATA_DOWNLOADS.md) for resource links. The configuration
example sets model paths and run options; it is not an environment report.

The repository keeps the main evidence only. Full console logs, per-round
debate traces, historical launch scripts, and package records remain backed up
on the local machine. The first repository commit also retains the earlier
full package; this cleanup does not rewrite that history.

Code is Apache-2.0 licensed; benchmark and Wikipedia text have separate terms
listed in [NOTICE.md](NOTICE.md), with article attribution links in the results.
No model weights, full corpus/index, or credentials are included.
AI assistance was used for tooling and documentation; results come from saved runs.
