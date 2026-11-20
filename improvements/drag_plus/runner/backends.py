"""Backend registry: the objects a run injects into the pipelines.

``fake`` is fully offline. ``local`` (``local_backend.build_local_backend``) talks to a
local Ollama server and replays stored passages; it is for smoke runs only.
``build_flashrag_backend`` imports FlashRAG, vLLM and transformers only when it is called,
mirroring ``scripts/run_evaluation.py``; it is not exercised on the laptop (see the
Server-unverified list in the pull request).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from ..instrument.timing import whitespace_tokens
from ..retrieval.cache import CachingRetriever, index_identity
from . import fake

METRICS = ["em", "f1", "acc", "precision", "recall"]
BACKEND_NAMES = ("fake", "flashrag", "local")


@dataclass
class Backend:
    name: str
    generator: Any
    retriever: Any
    make_template: Callable[[str, str], Any]
    evaluate: Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]]
    describe: Callable[[], Dict[str, Any]]
    retrieval_identity: Callable[[], Dict[str, Any]]
    count_tokens: Callable[[str], int] = whitespace_tokens

    def cached_retriever(self, path: Path, retriever: Any = None,
                         config: str = "") -> CachingRetriever:
        """One proxy per run, so both arms share the cache and the underlying calls."""
        return CachingRetriever(retriever if retriever is not None else self.retriever,
                                path=path, config=config, **self.retrieval_identity())


def build_retriever(flags: Any, dense: Any) -> Any:
    """Compose the run's retriever: the dense one, or the hybrid over it.

    The hybrid imports happen here, not at module import, so an ``e5`` run never loads
    the BM25 or reranker modules and ``bm25s`` is only imported on the first BM25 query.
    """
    if flags.retriever == "e5":
        return dense
    from ..retrieval.bm25 import Bm25Retriever
    from ..retrieval.hybrid import HybridRetriever
    from ..retrieval.rerank import CrossEncoderReranker
    if not flags.bm25_index:
        raise ValueError("retriever='hybrid' needs a bm25_index path")
    reranker = None if flags.rerank_k is None else CrossEncoderReranker(flags.rerank_model)
    return HybridRetriever.from_flags(flags, dense,
                                      Bm25Retriever(flags.bm25_index, depth=flags.bm25_top_k),
                                      reranker)


def _fake_identity() -> Dict[str, Any]:
    return {"index": "fake-retriever", "encoder": "fake", "top_k": 2}


def build_fake_backend() -> Backend:
    template = fake.FakeTemplate()
    return Backend("fake", fake.FakeGenerator(), fake.FakeRetriever(),
                   lambda arm, dataset: template, fake.fake_evaluate, fake.fake_description,
                   _fake_identity)


def _gpu_name() -> str:
    import torch  # lazy: only the real backend, only on the server
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def _flashrag_describe(overrides: Dict[str, Any]) -> Callable[[], Dict[str, Any]]:
    """The manifest's backend block: which GPU, model, index and seed the run used."""
    def describe() -> Dict[str, Any]:
        return {"gpu": _gpu_name(), "model": overrides["model2path"][overrides["generator_model"]],
                "index": overrides["index_path"], "corpus": overrides["corpus_path"],
                "seed": overrides.get("seed")}
    return describe


def _flashrag_identity(overrides: Dict[str, Any]) -> Callable[[], Dict[str, Any]]:
    """What a cache entry depends on besides the query: index, encoder, depth."""
    def identity() -> Dict[str, Any]:
        encoder = overrides["model2path"][overrides["retrieval_method"]]
        return {"index": index_identity(overrides["index_path"]), "encoder": encoder,
                "top_k": overrides.get("retrieval_topk")}
    return identity


def build_flashrag_backend(source: Path, overrides: Dict[str, Any], rows: List[dict],
                           out_dir: Path) -> Backend:
    """Build FlashRAG objects once, like the submission runner. Needs the A100 stack."""
    from flashrag.config import Config
    from flashrag.dataset import Dataset
    from flashrag.evaluator import Evaluator
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator, get_retriever
    from model.prompt import standard_rag_prompt_template
    from transformers import AutoTokenizer

    def config_for(dataset: str, split: str, arm: str) -> Any:
        settings = dict(overrides, dataset_name=dataset, split=[split], random_sample=False,
                        save_dir=str(out_dir / arm), save_note=f"{arm}_{dataset}",
                        metrics=METRICS, save_intermediate_data=True, save_metric_score=True)
        return Config(str(source / "config/base_config.yaml"), settings)

    first = rows[0]
    base = config_for(first["dataset"], first["split"], "drag")
    generator, retriever = get_generator(base), get_retriever(base)
    tokenizer = AutoTokenizer.from_pretrained(overrides["model2path"][overrides["generator_model"]])

    def make_template(arm: str, dataset: str) -> Any:
        config = config_for(dataset, first["split"], arm)
        return PromptTemplate(config) if arm == "drag" else standard_rag_prompt_template[dataset](config)

    def evaluate(arm: str, dataset: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        config = config_for(dataset, first["split"], arm)
        data = Dataset(config=config, data=[dict(
            id=r["id"], question=r["question"], golden_answers=r["golden_answers"],
            output={"pred": r["pred"]}) for r in records])
        return Evaluator(config).evaluate(data)

    return Backend("flashrag", generator, retriever, make_template, evaluate,
                   _flashrag_describe(overrides), _flashrag_identity(overrides),
                   lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
