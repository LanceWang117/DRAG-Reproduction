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
  `SHA-256("<seed>:<dataset>:<id>")` and take the first 100. This is deterministic
  across Python versions, unlike `random.sample` internals.
- **Exclusions:** dev identity `(dataset, split, id)`, any normalised question
  text already in the 200-question dev set, rows without a non-empty gold answer,
  and within-source duplicates by normalised text. Normalisation is NFKC,
  case-folding, keep alphanumerics, collapse whitespace.
- **Committed payload:** `dataset`, `split`, `id`, `question`, `golden_answers`,
  `position`. No passages or source metadata.
- **File size and hash:** 154,957 bytes, 600 lines, SHA-256
  `06b82f5808f4149829634aadf778e6385ed6b68cf886969e4b9ef35c1d3533cf`.

## Source splits (downloads are not committed)

Downloaded into the ignored `dataset/flashrag-bcafb8dd/` folder from
`https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/<revision>/<path>`.

| Dataset | HF path | Split | Rows | Eligible | Drawn | Bytes | SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2wiki | `2wikimultihopqa/dev.jsonl` | dev | 12,576 | 12,536 | 100 | 54,682,944 | `b4d74d73d1cb3c0632c665a85e1e5060e3411968b807721618cfd2464fc905e3` |
| HotpotQA | `hotpotqa/dev.jsonl` | dev | 7,405 | 7,370 | 100 | 48,191,374 | `434ec155867019396312f3d466ce3406c71fe8a2917f49f63bb646ec5ad2ff52` |
| NQ | `nq/test.jsonl` | test | 3,610 | 3,610 | 100 | 487,676 | `843bebb083064abb6eb23751f2c8c187e5c448c23900b42ca92f91502a9b7488` |
| PopQA | `popqa/test.jsonl` | test | 14,267 | 13,020 | 100 | 8,526,089 | `f976372fce3d7fe01b070357040391f10bd2710eeaa7c40d21c6574ffcad6cb8` |
| StrategyQA | `strategyqa/train.jsonl` | train | 2,290 | 2,264 | 100 | 1,867,298 | `4a2ee2ffc1af6336a52f34bc33aaa34dc25121daba3573d02e144ae3c4ccfdaa` |
| TriviaQA | `triviaqa/test.jsonl` | test | 11,313 | 11,310 | 100 | 4,797,585 | `8f0891dece17a6a6472c87bf0e8c555c80673853d8dc7780be4d47158ec85290` |

Total download: 118,442,966 bytes (about 113 MiB), under the 5 GB budget.

Why these splits: NQ and TriviaQA use the official `test` split because dev uses
their `dev` splits; PopQA publishes only `test`; StrategyQA publishes only
`train`; HotpotQA and 2wiki use their official `dev` split, with the dev rows
excluded, because their `train` splits are reserved for label draws (protocol
§1).

## Reproduce

```bash
mkdir -p dataset/flashrag-bcafb8dd
for path in nq/test.jsonl triviaqa/test.jsonl popqa/test.jsonl \
            hotpotqa/dev.jsonl 2wikimultihopqa/dev.jsonl strategyqa/train.jsonl; do
  mkdir -p "dataset/flashrag-bcafb8dd/$(dirname "$path")"
  curl -sSL "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/$path" \
    -o "dataset/flashrag-bcafb8dd/$path"
done
python scripts/draw_testset.py
sha256sum -c experiments/J05-testset/SHA256SUMS
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
