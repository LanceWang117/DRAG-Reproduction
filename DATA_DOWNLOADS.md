# Data and model downloads

Download large resources separately. The experiment used the full corpus and
index without rebuilding embeddings. See each resource's upstream access and
license terms.

| Resource | Source / version | Example local path |
| --- | --- | --- |
| Llama 3.1 8B Instruct | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), revision `0e9e39f249a16976918f6564b8830bc894c89659` | `models/Llama-3.1-8B-Instruct` |
| E5-base-v2 | [Hugging Face](https://huggingface.co/intfloat/e5-base-v2), revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | `models/e5-base-v2` |
| Corpus and index | [FlashRAG resources on ModelScope](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files), retrieval_corpus folder | `wiki_corpus/` |
| Full benchmark splits | [FlashRAG datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) | Not required to rerun our fixed inputs |

Llama requires your own approved Hugging Face account. Log in interactively;
never put a token into Git. From the repository root:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --local-dir models/Llama-3.1-8B-Instruct
huggingface-cli download intfloat/e5-base-v2 --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd --local-dir models/e5-base-v2
```

From ModelScope's retrieval_corpus folder, download `wiki18_100w.jsonl` and
`wiki18_100w_e5_index.zip`. Extract `e5_flat_inner.index` from the ZIP and put
it beside the JSONL under `wiki_corpus/`.

| File | Recorded size | SHA256 |
| --- | ---: | --- |
| wiki18_100w.jsonl | 14,393,573,105 bytes | `43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d` |
| wiki18_100w_e5_index.zip | 37,278,212,450 bytes | `8cdff95467e4544a60c4224f39a32df627b8f54d6124d83fa45fc28e36899b40` |
| Extracted index | 64,559,075,373 bytes | Size recorded; check downloaded ZIP hash above |

The corpus contains 21,015,324 passages of roughly 100 words, with 768-dimensional
E5 vectors. `100w` means 100 words, not one million documents. Each query
retrieves top-3 passages; DRAG's final pool can include multiple queries.

Our exact selected inputs are in `results/questions-200.jsonl`.
NQ, TriviaQA, HotpotQA, and 2Wiki use dev; PopQA uses test; StrategyQA uses train.
Other-five-dataset downloads used revision
`bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`; NQ was downloaded separately
from the then-current main branch. Full download manifests remain in the local
archive. The saved inputs let readers reuse the sample without resampling.

Benchmark questions and Wikipedia excerpts retain upstream rights; the code
license does not relicense their content. Passage titles/IDs are retained in
the retrieval records. Review upstream terms before publishing those records.
