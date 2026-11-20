"""Backend registry: the objects a run injects into the pipelines.

``fake`` is fully offline. ``build_flashrag_backend`` imports FlashRAG, vLLM and
transformers only when it is called, mirroring ``scripts/run_evaluation.py``; it is not
exercised on the laptop (see the Server-unverified list in the pull request).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from ..instrument.timing import whitespace_tokens
from . import fake

METRICS = ["em", "f1", "acc", "precision", "recall"]


@dataclass
class Backend:
    name: str
    generator: Any
    retriever: Any
    make_template: Callable[[str, str], Any]
    evaluate: Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]]
    describe: Callable[[], Dict[str, Any]]
    count_tokens: Callable[[str], int] = whitespace_tokens


def build_fake_backend() -> Backend:
    template = fake.FakeTemplate()
    return Backend("fake", fake.FakeGenerator(), fake.FakeRetriever(),
                   lambda arm, dataset: template, fake.fake_evaluate, fake.fake_description)


def _gpu_name() -> str:
    import torch  # lazy: only the real backend, only on the server
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


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

    def describe() -> Dict[str, Any]:
        return {"gpu": _gpu_name(), "model": overrides["model2path"][overrides["generator_model"]],
                "index": overrides["index_path"], "corpus": overrides["corpus_path"]}

    return Backend("flashrag", generator, retriever, make_template, evaluate, describe,
                   lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
