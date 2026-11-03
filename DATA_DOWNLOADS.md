# Models, corpus, and question sets

Large resources are downloaded separately. No embedding or index rebuild was
performed for the reported experiments.

## Models

| Resource | Revision used | Local directory for the example configuration |
| --- | --- | --- |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659` | `models/Llama-3.1-8B-Instruct` |
| [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | `models/e5-base-v2` |

Llama requires approval of the model's access terms using your own Hugging Face
account. Authenticate interactively; do not place a token in a script, command
argument, screenshot, or Git commit.

From the repository root, in the Linux environment:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --local-dir models/Llama-3.1-8B-Instruct
huggingface-cli download intfloat/e5-base-v2 \
  --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
  --local-dir models/e5-base-v2
```

These commands download the pinned snapshots, which may include auxiliary
files. The experiment used BF16 Llama weights, not the Mac's Ollama quantization.

## Wikipedia corpus and E5 index

The authors link to [FlashRAG retrieval resources on ModelScope](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files).
Download these files from `retrieval_corpus`:

| File | Recorded size | SHA256 |
| --- | ---: | --- |
| `wiki18_100w.jsonl` | 14,393,573,105 bytes | `43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d` |
| `wiki18_100w_e5_index.zip` | 37,278,212,450 bytes | `8cdff95467e4544a60c4224f39a32df627b8f54d6124d83fa45fc28e36899b40` |

Extract `e5_flat_inner.index` from the ZIP. Its recorded size is 64,559,075,373
bytes. Place the JSONL and extracted index under `wiki_corpus/` at the repository
root, or update `config.local.json` to point to existing copies. Allow additional
disk space for the ZIP during setup.

The corpus contains 21,015,324 passages of roughly 100 words. `100w` means
100-word chunks, not one million articles. The JSONL stores text and the FAISS
FlatIP index stores 768-dimensional E5 vectors. Each query retrieves top-3
passages. DRAG can issue several queries, so its final evidence pool can contain
more than three passages overall.

Historical download URLs and checks are in `results/retrieval-resource-manifest.json`
and `scripts/cloud_fetch_resources.py`. The historical downloader assumes the
old `/workspace/Capstone_project` layout; do not run it blindly in this checkout.

## Evaluation questions

Source: [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).
The original full splits are not needed to inspect or rerun our fixed 200
questions: the small selected inputs are already saved under `results/`.

| Dataset | Source path | Split used | Selected questions |
| --- | --- | --- | ---: |
| NQ | `nq/dev.jsonl` | dev | 50 |
| TriviaQA | `triviaqa/dev.jsonl` | dev | 10 |
| PopQA | `popqa/test.jsonl` | test | 40 |
| 2Wiki | `2wikimultihopqa/dev.jsonl` | dev | 40 |
| HotpotQA | `hotpotqa/dev.jsonl` | dev | 35 |
| StrategyQA | `strategyqa/train.jsonl` | train | 25 |

The other-five-dataset download revision was
`bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`. Exact split URLs, counts, and checksums
are in `results/benchmark-download-manifest.json`. NQ was fetched separately
from the repository's then-current `main`; its downloaded content checksum is
preserved in the run manifests rather than claiming a pinned revision.

The complete selected list is
`results/20261017T033818Z_paired_extension100/question-manifest-200.jsonl`.
Rows include `dataset`, `split`, `id`, `question`, and `golden_answers`.

## Attribution and redistribution

The FlashRAG repository preprocesses datasets from separate benchmark projects;
it does not make them a single newly licensed dataset. Retain the source
references and consult each upstream dataset's terms before redistribution.
The included question subsets and retrieved Wikipedia text are for this course
reproduction record. They are not covered by the code's Apache-2.0 license.
Wikipedia text attribution is retained through passage titles/IDs and the
corpus provenance above. Consult the applicable Wikipedia content terms before
republishing excerpts. Llama and E5 retain their own model licenses.
