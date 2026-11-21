# Pre-registration: held-out test set for DRAG+ evaluation

This document fixes the evaluation plan before the test questions are drawn or
run. The git commit that first adds this file is the pre-registration timestamp;
its SHA is recorded in the job status log. The draw is seeded and reproducible
from the parameters below. The drawn bytes are sealed by `SHA256SUMS`, committed
with the draw.

## 1. Evaluation question and arms

- **Confirmatory question.** For each frozen configuration, does pooled exact
  match (EM) on the 600 held-out questions differ from the frozen baseline?
- **Pairing.** Every comparison uses the same 600 questions, the same retrieval
  cache, the same answer extractor, the same commit and the same environment
  (protocol §2). The baseline arm is the baseline rerun by E00 with the current
  runner, never the historical `results/` numbers.
- **Frozen configurations.** The configurations to test are frozen in
  `experiments/E06/FROZEN.md` before the test run. After the test set has been
  run, no configuration, prompt, threshold or code change can be presented as
  confirmatory.

## 2. Test set

- **File.** `experiments/J05-testset/questions-test.jsonl`, 600 rows, 100 per
  dataset: 2Wiki, HotpotQA, NQ, PopQA, StrategyQA and TriviaQA. Each row holds
  only `dataset`, `split`, `id`, `question`, `golden_answers` and `position`.
- **Sources.** `RUC-NLPIR/FlashRAG_datasets` at revision
  `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`:

  | Dataset | Source split | Rationale |
  | --- | --- | --- |
  | 2Wiki | `2wikimultihopqa/dev` | train is reserved for label draws |
  | HotpotQA | `hotpotqa/dev` | train is reserved for label draws |
  | NQ | `nq/test` | dev is already used |
  | PopQA | `popqa/test` | only split published |
  | StrategyQA | `strategyqa/train` | only split published |
  | TriviaQA | `triviaqa/test` | dev is already used |

- **Seed and method.** Seed `20261121`. Within each dataset, eligible rows are
  ranked by ascending `SHA-256("<seed>:<dataset>:<id>")` and the first 100 are
  taken. The rule does not depend on Python's `random` implementation, so a
  re-draw at the pinned revision reproduces the file byte for byte.
- **Disjointness.** A source row is ineligible if its `(dataset, split, id)` is
  in the 200-question dev set, if its normalised question text equals any dev
  question, if it has no non-empty gold answer, or if its normalised text
  duplicates an already-selected candidate. Normalisation is NFKC, case-folding,
  keep alphanumerics, collapse whitespace.
- **Sealing.** These 600 questions must not be used for prompting, few-shot
  examples, threshold tuning, error inspection or any other model-visible
  purpose before E06. If leakage is discovered, it is disclosed and the affected
  comparison is reported as exploratory.

## 3. Primary endpoint

- **Metric.** Pooled EM over all 600 questions, for each pre-registered
  configuration against the frozen baseline.
- **Test.** Exact two-sided McNemar test on the discordant pairs, α = 0.05.
- **Scoring.** EM is produced by the committed scorer,
  `improvements/drag_plus/analysis/score.py::exact_match`: normalise the
  prediction and every gold answer, then score 1 if any matches. No hand
  scoring.
- **Decision.** A comparison is significant when p < 0.05. Report the discordant
  counts (b, c), ΔEM with its 95% bootstrap CI, and the p-value.

## 4. Secondary endpoints

- **Per-dataset EM.** Exact McNemar per dataset, with Holm–Bonferroni across the
  six datasets at family α = 0.05.
- **F1.** Paired bootstrap over questions, 10,000 resamples, 95% percentile CI,
  seed 2026 (`improvements/drag_plus/analysis/stats.py::paired_bootstrap_ci`),
  pooled and per dataset.
- **Latency.** Per-question paired seconds; median and p90 with bootstrap CIs;
  two-sided Wilcoxon signed-rank test at α = 0.05, pooled.
- **Descriptive only.** `acc`, `Recall@k`, call counts and token counts are
  reported per protocol §5 but are not confirmatory tests.

## 5. Sample size and power

The protocol §4 formula gives n ≈ 590 paired questions to detect δ = 5 pp at
ψ ≈ 0.19 discordant fraction, two-sided α = 0.05 and 80% power. The draw is 600,
so the planned minimum detectable effect is about 5 pp. The observed discordant
fraction and the exact MDE are reported with every result.

## 6. One run per configuration and stopping rules

- **Exactly one test run per frozen configuration.** No interim analysis, no
  peeking at partial outputs and no early stopping on results.
- **All 600 questions are run.** A run that completes fewer than 600 questions
  for infrastructure reasons is incomplete: it is not analysed on the completed
  subset. Fix the infrastructure, resume from the J04 retrieval cache under the
  same frozen configuration, and rerun all 600.
- **Scoring failures are data.** An empty or unparseable answer scores EM = 0 and
  F1 = 0. It is never dropped or retried with a different prompt.
- Wall-clock or throughput measurements (protocol §4) are never a reason to stop
  a run.

## 7. No post-hoc exclusions or changes

- No question is excluded after seeing outputs. No new configuration, prompt
  edit, threshold change or metric substitution is added after the first test
  run.
- A deviation forced by a bug is recorded in a new commit stating what changed
  and why, before the affected results are interpreted. If it is discovered only
  after seeing results, that comparison is reported as exploratory, not
  confirmatory.
- The primary endpoint is never swapped for a secondary one.

## 8. Reporting of null and negative results

Every pre-registered configuration is reported, including null and negative
results, with the same metrics and detail as a positive one (protocol §5). Each
report includes EM, F1, acc and Recall@k; calls and tokens per question; median
and p90 seconds per question and questions/hour; startup time; peak VRAM; the
A/A floor; the discordant counts; all p-values and CIs; and the MDE. Figures come
from committed Matplotlib scripts.

## 9. Amendments

This document is append-only. An amendment is a new dated section added in a
commit made before the test run; it never edits sections 1–8. The original
commit remains the pre-registration timestamp.
