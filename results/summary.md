# Evaluation record

| File | Purpose |
| --- | --- |
| `questions-200.jsonl` | Exact question IDs, text, reference answers, and splits |
| `paired-200.json` | Both methods' answers, per-question metrics, and dataset totals |
| `retrieval-200.jsonl` | Both methods' retrieved passage IDs and full text; one row per question |
| `summary.csv` | Six-dataset EM/F1 table for spreadsheets |
| `run-records.json` | Four completed run summaries, timing, exit status, and resource peaks |
| `checksums.json` | SHA256 checks for the data files and attribution list |
| `passage-sources.csv` | Wikipedia article/contributor links for saved passage titles |

DRAG minus Naive EM, in percentage points:

| Dataset | Paper | Our sample |
| --- | ---: | ---: |
| NQ | -1.40 | -10.00 |
| TriviaQA | 0.00 | +10.00 |
| PopQA | +1.00 | +10.00 |
| 2Wiki | +14.00 | +20.00 |
| HotpotQA | +5.00 | +5.71 |
| StrategyQA | +6.60 | 0.00 |

Source: [paper Table 1](https://aclanthology.org/2025.acl-long.770.pdf).
Across this sample: Naive EM/F1 39.00/44.96; DRAG 44.00/52.06.
Mean serial inference time: Naive 4.86s; DRAG 28.23s, excluding initialization.
DRAG alone matched exactly on 24 questions; Naive alone on 14; 162 tied in EM.

This was targeted expansion from 100 to 200 questions, not a balanced test.
StrategyQA uses train; DRAG includes the Judge fix. Scores are not directly
comparable with the paper's 500 questions per dataset. EM measures normalized
exact matching; F1 is a token-overlap score, not a pass percentage.

These compact records were extracted from the original saved runs without
rewriting answers or scores. Full console logs, debate traces, checkpoints,
resource samples, and original download manifests remain in the local archive
and the first repository commit. The GitHub history was not rewritten.
