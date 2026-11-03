# Sources and reuse notices

The code license and the data licenses are different.

- DRAG source and figures: [Huenao/Debate-Augmented-RAG](https://github.com/Huenao/Debate-Augmented-RAG),
  Apache-2.0. The original LICENSE and README are retained in `src/original_drag/`.
- Processed benchmark questions/reference answers: [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).
  Its dataset card declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  We selected subsets, retained IDs/splits, and added predictions and scores.
  This does not turn the underlying dataset text into Apache-licensed code.
- Retrieved text: Wikipedia contributors, through the FlashRAG `wiki18_100w`
  corpus. The text was split into roughly 100-word passages upstream. We selected
  passages for retrieval records without rewriting their text. Wikipedia text
  remains under its [applicable content licenses](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content),
  including [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/), not Apache-2.0.
  Article and contributor-history links are listed in `results/passage-sources.csv`.
  They identify sources, not the exact revision of the historical corpus.

Original benchmark projects: [NQ](https://github.com/google-research-datasets/natural-questions),
[TriviaQA](https://nlp.cs.washington.edu/triviaqa/),
[PopQA](https://github.com/AlexTMallen/adaptive-retrieval),
[HotpotQA](https://hotpotqa.github.io/),
[2Wiki](https://github.com/Alab-NII/2wikimultihop), and
[StrategyQA](https://github.com/eladsegal/strategyqa). Their underlying material
retains any additional upstream notices and terms.

These notices also apply to copies of the same third-party material in earlier
repository commits. The passage-source list includes titles found in those
saved records. No model weights or full corpus/index are redistributed.
