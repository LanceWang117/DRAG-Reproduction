# Saved experiment evidence

`fixed200-drag-naive-comparison.json` is the easiest starting point. It contains
all 200 paired questions, reference answers, both predictions, per-question
metrics, and available individual inference times. A question is identified by
dataset, split, and ID together; IDs alone can repeat across datasets.

The original run folders are kept without rewriting their file contents:

| Folder | Used in the final 200-question comparison |
| --- | --- |
| `20261017T010415Z_drag_50q` | Fixed DRAG v1.0.1, NQ 50 |
| `20261017T015401Z_drag_extra50_v1.0.1` | Fixed DRAG, 10 questions from each other dataset |
| `20261017T030736Z_naive_rag_fixed100` | Naive, the same first 100 questions |
| `20261017T033818Z_paired_extension100` | Both methods, the same 100 additional questions |

The extension's `question-manifest-200.jsonl` lists the complete fixed sample.
The saved source subsets appear under each run's `fixed-datasets` (or
`fixed-dataset` for NQ) directory. These are small selected subsets, not full
benchmark downloads.

Within the run folders:

- `intermediate_data.json`: questions, gold answers, predictions, metrics, and
  intermediate outputs. Find it below `official-results`.
- Naive `output.retrieval_result`: three retrieved passages with IDs and text.
- DRAG `output.QueryStage_QueryPool`: final queries and their top-3 passages.
  Per-round prompt/output fields also preserve the debate trace.
- `question-checkpoints.jsonl`: per-question durable checkpoints where available.
- `console.log` and `supervisor.log`: captured process output.
- `resources.jsonl`: sampled process memory, cgroup memory, and GPU readings.
- `run-summary.json`: exit status, timeout flag, and supervisor wall time.
- `dataset-results.json` and `metric_score.txt`: dataset metrics and timing.

Raw files contain historical absolute Linux paths. They describe the machine
used at the time, not required paths on the reader's computer. Folder names use
UTC, which can differ from the Vancouver date by one day.

The `*-backup-verification.json` records and original manifests retain earlier
SHA256 checks. `checksums.json` separately checks every copied run file in this
repository. `scripts/verify_evidence.py` validates those hashes and the paired
identities without loading the model. No GPU or network is required.

The upstream Apache-2.0 license covers code, not all question or passage text.
Benchmark subsets and retrieved passages retain their upstream rights and
attribution; see `DATA_DOWNLOADS.md`. This repository is initially private for
course review. Do not publish the saved text without reviewing those terms.
