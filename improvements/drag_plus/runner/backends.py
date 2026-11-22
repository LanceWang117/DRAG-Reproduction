"""Backend registry: the objects a run injects into the pipelines.

``fake`` is fully offline. ``local`` (``local_backend.build_local_backend``) talks to a
local Ollama server and replays stored passages; it is for smoke runs only.
``build_flashrag_backend`` imports FlashRAG, vLLM and transformers only when it is called,
mirroring ``scripts/run_evaluation.py``; it is not exercised on the laptop (see the
Server-unverified list in the pull request).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional

from ..instrument.timing import whitespace_tokens
from ..retrieval.cache import CachingRetriever, index_identity
from . import fake

METRICS = ["em", "f1", "acc", "precision", "recall"]
BACKEND_NAMES = ("fake", "flashrag", "local")
FAISS_GPU_HELP = ("faiss_gpu=true needs FAISS with GPU support; install the pinned "
                  "faiss-gpu build (see REPRODUCTION.md, 'GPU retrieval')")


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


def apply_thread_environment(flags: Any,
                             environ: Optional[MutableMapping[str, str]] = None
                             ) -> MutableMapping[str, str]:
    """Set OpenMP/MKL thread counts before FAISS, torch or FlashRAG is imported.

    FAISS reads ``OMP_NUM_THREADS`` at its OpenMP runtime and MKL reads
    ``MKL_NUM_THREADS`` when its shared library loads; setting them afterwards can be
    ignored. The variant is authoritative, so an inherited value is replaced. The same
    call works for ``faiss_gpu=false``, where only the CPU scan changes.
    """
    target = os.environ if environ is None else environ
    threads = str(flags.omp_threads)
    target["OMP_NUM_THREADS"] = threads
    target["MKL_NUM_THREADS"] = threads
    return target


def ensure_faiss_gpu(faiss_module: Any) -> None:
    """Fail early when FAISS has no GPU support or sees no GPU; never fall back to CPU."""
    if not hasattr(faiss_module, "index_cpu_to_all_gpus"):
        raise RuntimeError(FAISS_GPU_HELP)
    try:
        gpu_count = faiss_module.get_num_gpus()
    except Exception as error:  # a driver/runtime error is a configuration error here
        raise RuntimeError(f"{FAISS_GPU_HELP}: {error}") from error
    if gpu_count < 1:
        raise RuntimeError(f"{FAISS_GPU_HELP}: no CUDA GPU is visible")


def _require_faiss(flags: Any) -> None:
    """Import FAISS (after the thread env) and refuse a CPU-only build for faiss_gpu."""
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is not installed; see REPRODUCTION.md, 'GPU retrieval'") from error
    if flags.faiss_gpu:
        ensure_faiss_gpu(faiss)


def build_rag_objects(config: Any, faiss_gpu: bool, generator_factory: Callable[[Any], Any],
                      retriever_factory: Callable[[Any], Any]) -> tuple:
    """Build the retriever before the generator when the index moves to the GPU.

    ``DenseRetriever`` places the fp16 index on the GPUs in its constructor, while
    FlashRAG's vLLM generator reserves its KV pool in its constructor. Building the
    generator first would reserve that pool before the index clone and can exhaust the
    card. The CPU path keeps the recorded baseline order: generator first.
    """
    if faiss_gpu:
        retriever = retriever_factory(config)
        generator = generator_factory(config)
    else:
        generator = generator_factory(config)
        retriever = retriever_factory(config)
    return generator, retriever


def flashrag_settings(overrides: Dict[str, Any], flags: Any, dataset: str, split: str,
                      arm: str, out_dir: Path) -> Dict[str, Any]:
    """The Config override dict for one (arm, dataset): variant flags win over config.

    ``faiss_gpu`` and ``gpu_memory_utilization`` are always explicit so the run's
    manifest and trace show the values the backend actually used.
    """
    return dict(overrides, dataset_name=dataset, split=[split], random_sample=False,
                save_dir=str(out_dir / arm), save_note=f"{arm}_{dataset}",
                metrics=METRICS, save_intermediate_data=True, save_metric_score=True,
                faiss_gpu=flags.faiss_gpu,
                gpu_memory_utilization=flags.gpu_memory_utilization)


def _vllm_llm(model_path: str, **kwargs: Any) -> Any:
    from vllm import LLM  # lazy: only the real backend
    return LLM(model_path, **kwargs)


def _auto_tokenizer(model_path: str, **kwargs: Any) -> Any:
    from transformers import AutoTokenizer  # lazy: only the real backend
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def build_generator(config: Any, flags: Any, get_generator: Callable[[Any], Any],
                    base_cls: Any = None, base_generator_cls: Any = None,
                    llm_factory: Optional[Callable[..., Any]] = None,
                    tokenizer_factory: Optional[Callable[..., Any]] = None) -> Any:
    """FlashRAG's generator, or an APC-enabled VLLMGenerator subclass when flagged.

    FlashRAG builds ``vllm.LLM`` inline with no ``enable_prefix_caching`` hook, so the
    flag subclasses it with the same arguments; off calls the factory unchanged.
    """
    if not flags.enable_prefix_caching:
        return get_generator(config)
    if base_cls is None or base_generator_cls is None:
        raise ValueError("enable_prefix_caching needs the FlashRAG VLLMGenerator classes")
    make_llm, make_tokenizer = llm_factory or _vllm_llm, tokenizer_factory or _auto_tokenizer

    class PrefixCachingVLLMGenerator(base_cls):
        def __init__(self, config: Any):
            base_generator_cls.__init__(self, config)
            options = dict(tensor_parallel_size=self.tensor_parallel_size,
                           gpu_memory_utilization=self.gpu_memory_utilization,
                           max_logprobs=32016, max_model_len=self.max_model_len,
                           enable_prefix_caching=True)
            if self.use_lora:
                options.update(enable_lora=True, max_lora_rank=64)
            self.model = make_llm(self.model_path, **options)
            self.tokenizer = make_tokenizer(self.model_path, trust_remote_code=True)

    return PrefixCachingVLLMGenerator(config)


def build_flashrag_backend(source: Path, overrides: Dict[str, Any], rows: List[dict],
                           out_dir: Path, flags: Any) -> Backend:
    """Build FlashRAG objects once, like the submission runner. Needs the A100 stack."""
    apply_thread_environment(flags)  # before Config/torch/faiss import, not after
    _require_faiss(flags)
    from flashrag.config import Config
    from flashrag.dataset import Dataset
    from flashrag.evaluator import Evaluator
    from flashrag.generator.generator import BaseGenerator, VLLMGenerator
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator, get_retriever
    from model.prompt import standard_rag_prompt_template
    from transformers import AutoTokenizer

    def config_for(dataset: str, split: str, arm: str) -> Any:
        settings = flashrag_settings(overrides, flags, dataset, split, arm, out_dir)
        return Config(str(source / "config/base_config.yaml"), settings)

    first = rows[0]
    base = config_for(first["dataset"], first["split"], "drag")
    def generator_factory(config: Any) -> Any:
        return build_generator(config, flags, get_generator, VLLMGenerator, BaseGenerator)

    generator, retriever = build_rag_objects(base, flags.faiss_gpu, generator_factory,
                                             get_retriever)
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
