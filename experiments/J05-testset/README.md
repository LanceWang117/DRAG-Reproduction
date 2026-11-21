# J05: held-out test set

This folder holds the pre-registered, 600-question held-out evaluation set for
DRAG+ (protocol §1). It was drawn from the FlashRAG benchmark splits **before any
model saw the questions**, with a committed seed and checksums. Do not use these
questions for prompting, few-shot examples, tuning or error inspection before
E06.

## Files

| File | Purpose |
| --- | --- |
| `questions-test.jsonl` | 600 test questions: identity, question, gold answers, position |
| `PREREGISTRATION.md` | Endpoints, tests, α, sample size, stopping and reporting rules |
| `SHA256SUMS` | SHA-256 of this README, the pre-registration and the question file |
| `README.md` | This record |

## Draw parameters

- **Source revision:** `RUC-NLPIR/FlashRAG_datasets` at
  `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`. That commit is also the latest
  commit on the dataset's main branch (2025-05-06), so all six datasets use one
  pinned revision, including NQ.
- **Seed:** `20261121`.
- **Allocation:** 100 questions per dataset, 600 total. No shortfall: the
  smallest eligible pool is StrategyQA with 2,264 rows.
- **Selection rule:** within each dataset, rank eligible rows by ascending
  `SHA-256("<seed>:<dataset>:<id>")`, then scan the ranked rows and keep the
  first row of each unseen normalised question text until 100 are selected. The
  scan continues across datasets in the fixed order 2wiki, HotpotQA, NQ, PopQA,
  StrategyQA, TriviaQA, so no question is selected twice. This is deterministic
  across Python versions, unlike `random.sample` internals.
- **Output order:** rows are emitted dataset by dataset in the order above and
  within a dataset in rank order; `position` is the 0-based index in that order.
- **Exclusions:** dev identity `(dataset, split, id)`, any normalised question
  text already in the 200-question dev set, rows without a non-empty gold answer,
  and any normalised question text already selected. Normalisation is NFKC,
  case-folding, keep alphanumerics, collapse whitespace.
- **Committed payload:** `dataset`, `split`, `id`, `question`, `golden_answers`,
  `position`. No passages or source metadata.
- **File size and hash:** 155,263 bytes, 600 lines, SHA-256
  `5d76fbe39c36e7b15a3bd4330e6909429f455ff4e885e79a2a0496c2ac338a38`.
- **Interpreter:** the same bytes were reproduced under CPython 3.11.18
  (Unicode 14.0, the gate's interpreter) and CPython 3.14.10 (Unicode 16.0).

## Source splits (downloads are not committed)

Downloaded into the ignored `dataset/flashrag-bcafb8dd/` folder from
`https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/<revision>/<path>`.

| Dataset | HF path | Split | Rows | Distinct | Drawn | Bytes | SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2wiki | `2wikimultihopqa/dev.jsonl` | dev | 12,576 | 12,536 | 100 | 54,682,944 | `b4d74d73d1cb3c0632c665a85e1e5060e3411968b807721618cfd2464fc905e3` |
| HotpotQA | `hotpotqa/dev.jsonl` | dev | 7,405 | 7,370 | 100 | 48,191,374 | `434ec155867019396312f3d466ce3406c71fe8a2917f49f63bb646ec5ad2ff52` |
| NQ | `nq/test.jsonl` | test | 3,610 | 3,610 | 100 | 487,676 | `843bebb083064abb6eb23751f2c8c187e5c448c23900b42ca92f91502a9b7488` |
| PopQA | `popqa/test.jsonl` | test | 14,267 | 13,020 | 100 | 8,526,089 | `f976372fce3d7fe01b070357040391f10bd2710eeaa7c40d21c6574ffcad6cb8` |
| StrategyQA | `strategyqa/train.jsonl` | train | 2,290 | 2,264 | 100 | 1,867,298 | `4a2ee2ffc1af6336a52f34bc33aaa34dc25121daba3573d02e144ae3c4ccfdaa` |
| TriviaQA | `triviaqa/test.jsonl` | test | 11,313 | 11,310 | 100 | 4,797,585 | `8f0891dece17a6a6472c87bf0e8c555c80673853d8dc7780be4d47158ec85290` |

"Distinct" counts unique normalised question texts after the dev-identity,
dev-text and gold-answer filters. Removed at that stage: 2wiki 40 dev ids;
HotpotQA 35 dev ids; StrategyQA 25 dev ids and 1 internal duplicate; TriviaQA 3
internal duplicates; PopQA 40 dev ids, 10 dev-text hits and 1,197 internal
duplicate rows; NQ nothing beyond its 3,610 rows.

Total download: 118,552,966 bytes (about 113 MiB), under the 5 GB budget.

Why these splits: NQ and TriviaQA use the official `test` split because dev uses
their `dev` splits; PopQA publishes only `test`; StrategyQA publishes only
`train`; HotpotQA and 2wiki use their official `dev` split, with the dev rows
excluded, because their `train` splits are reserved for label draws (protocol
§1).

Known limitation: 565 PopQA duplicate-text groups (and one TriviaQA group) carry
different reference answers across their rows. The seeded rank picks the group's
highest-ranked row, including its answers; 8 of the 600 selected rows (all
PopQA) come from such groups. Amendment A1 records this and it is not changed
after seeing outputs.

## Reproduce

```bash
mkdir -p dataset/flashrag-bcafb8dd
for path in nq/test.jsonl triviaqa/test.jsonl popqa/test.jsonl \
            hotpotqa/dev.jsonl 2wikimultihopqa/dev.jsonl strategyqa/train.jsonl; do
  mkdir -p "dataset/flashrag-bcafb8dd/$(dirname "$path")"
  curl -fsSL "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/$path" \
    -o "dataset/flashrag-bcafb8dd/$path"
done
# Confirm each download against the SHA-256 column above before drawing.
python scripts/draw_testset.py
(cd experiments/J05-testset && sha256sum -c SHA256SUMS)
```

The script prints only per-dataset counts, the byte count and the output hash;
it never prints questions. A re-draw at the pinned revision and seed reproduces
`questions-test.jsonl` byte for byte.

## Verification

`improvements/tests/test_draw_testset.py` checks, on every CI run, that the
committed file has 600 unique rows at 100 per dataset, contains no dev identity
or normalised-text overlap, contains no duplicate question, stays under 1 MB, and
matches every entry in `SHA256SUMS`. Unit tests cover dev exclusion, missing
answers, deduplication, shortfall reporting and byte-identical re-draws on
synthetic fixtures.
