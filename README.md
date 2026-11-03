# DRAG reproduction

This repository contains our capstone reproduction of Debate-Augmented RAG
(DRAG). We compared DRAG and Naive RAG on the same 200 questions, using
Llama 3.1 8B Instruct and the full Wikipedia corpus and E5 index.

The experiments ran on a RunPod A100 instance in October 2026. This is a
small-scale reproduction, not a rerun of the paper's full evaluation. We also
fixed an ambiguous retrieval Judge output check. The reported DRAG results use
that fix; they are not results from unchanged upstream code.

## Results

Scores below are percentages. Each row uses the same questions for both methods.

| Dataset | Questions | Naive EM | Naive F1 | DRAG EM | DRAG F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NQ | 50 | 44.00 | 54.72 | 34.00 | 50.56 |
| TriviaQA | 10 | 60.00 | 67.25 | 70.00 | 72.86 |
| PopQA | 40 | 45.00 | 49.47 | 55.00 | 59.76 |
| 2Wiki | 40 | 15.00 | 18.00 | 35.00 | 38.13 |
| HotpotQA | 35 | 28.57 | 36.67 | 34.29 | 46.84 |
| StrategyQA | 25 | 64.00 | 64.00 | 64.00 | 64.00 |

Across this particular 200-question mixture, EM was 44% for DRAG and 39% for
Naive RAG. Average inference time was about 28.23 seconds and 4.86 seconds per
question, respectively, excluding initialization. These are serial runs, not
maximum-throughput serving benchmarks.

The gains on 2Wiki and HotpotQA follow the direction reported in the paper.
DRAG performed worse on NQ and tied on StrategyQA. The sample is too small and
uneven to claim full reproduction or statistically established improvement.

## Files

- `src/`: complete source snapshot with the Judge fix, original configuration,
  upstream README, and tests. Original imports and directory structure are kept.
- `scripts/`: the scripts used for the cloud experiments, plus a portable runner
  and an offline evidence checker added for this submission.
- `results/`: paired question-level results, fixed question lists, saved runs,
  metrics, resource samples, and download manifests.
- [REPRODUCTION.md](REPRODUCTION.md): setup and execution commands.
- [DATA_DOWNLOADS.md](DATA_DOWNLOADS.md): resources, versions, and download links.
- [ENVIRONMENT.md](ENVIRONMENT.md): hardware, software, and local Mac trial.
- [JUDGE_FIX.md](JUDGE_FIX.md): defect, repair policy, and verification.
- [CONTRIBUTIONS.md](CONTRIBUTIONS.md): recorded project contributions.

For a quick inspection without a GPU:

```bash
python3 scripts/verify_evidence.py
python3 scripts/run_evaluation.py --validate-only
```

Read `results/summary.md` for the paper comparison. Individual answers are in
`results/fixed200-drag-naive-comparison.json`. The complete saved evidence is
described in `results/README.md`. Models, full datasets, corpus, index, and
credentials are deliberately excluded from Git.

## Scope and limitations

- The first pilot used 50 NQ questions and 10 from each other dataset. We then
  added 100 questions to selected datasets after reviewing the pilot. This was
  targeted expansion, not a balanced independent confirmation experiment.
- StrategyQA uses 25 questions from the available training split. We did not
  train a model, but this is not a held-out evaluation or the paper's protocol.
- The paper sampled 500 questions per dataset. Our question IDs need not match
  the paper's sample, so absolute scores are not directly comparable.
- EM and token F1 measure answer matching, not all factual errors or user
  satisfaction. No additional LLM judge was used to calculate these metrics.
- The original NQ DRAG run has a batch inference time but no exact individual
  question times. Those fields remain null in the paired results.
- The Judge fix can add retry calls. It is a validation fix, not a demonstrated
  accuracy improvement or the project's proposed new method.

Next work is to examine failure cases, study the debate code, and consider ways
to avoid unnecessary debate. No routing improvement is claimed here.

## Sources and license

Paper: [DRAG, ACL 2025](https://aclanthology.org/2025.acl-long.770/).
Original code: [Huenao/Debate-Augmented-RAG](https://github.com/Huenao/Debate-Augmented-RAG),
baseline commit `c035ee1b7e6bc9fa67d5a104df1c2175a8300666`.
Our fixed source snapshot: `3870c0ea3b675da63e4bc8047af07604073568f1` in the local
experiment history. These identifiers refer to the source history, not to this
submission repository's initial commit.

The original code is Apache-2.0 licensed. Its license is retained in `LICENSE`
and `src/LICENSE`; our added Python scripts are provided under the same license.
Model weights and dataset content remain subject to their respective upstream
terms. The repository license does not relicense those resources or retrieved
Wikipedia excerpts. Keep these notices if redistributing the saved evidence.

We used AI assistance for setup, scripting, and documentation. The uploaded
results come from the saved runs; the evidence checker reads them without
calling a model or changing scores. See `CONTRIBUTIONS.md` for scope.
