"""Automatic server probes for E00 (J26): seconds each, no full question.

Each probe takes a ``ProbeContext`` (the objects a real run builds, injected) and returns
``(ok, detail)``. Probes marked ``server`` need the FlashRAG/vLLM/FAISS stack and are
skipped on the fake and local backends; the others also run on the PC, which is how the
runner itself is tested. Server modules are imported inside the probes, never here.
"""
from __future__ import annotations

import importlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..gates.escalation import SUFFICIENCY_LABELS, sufficiency_messages
from ..judge.logprob import LABELS, SCORED, margin
from ..retrieval.cache import index_identity

Result = Tuple[bool, str]
JUDGE_SYSTEM = ("You are the judge in a debate about whether the retrieved passages answer "
                "the question. Proponent argued: {pro}. Opponent argued: {opp}. "
                "Output only the agent's name: Proponent or Opponent. No explanations.")
DEBATES = (("the passages name the answer directly", "the passages are about a different person"),
           ("two passages agree on the date", "the date is missing from every passage"),
           ("the title passage states the fact", "the query needs the year, not the place"))
ORDER_WORDS = ("apple", "river", "violet", "marble")


@dataclass
class ProbeContext:
    """What the probes may touch; a test injects fakes for every field."""

    backend: str
    generator: Any
    retriever: Any
    make_template: Callable[[str, str], Any]
    flags: Any
    stored: Sequence[Tuple[str, List[str], List[dict]]] = ()  # question, naive ids, docs
    index_path: Optional[str] = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    import_module: Callable[[str], Any] = importlib.import_module
    read_gpu_gib: Optional[Callable[[], float]] = None
    clock: Callable[[], float] = time.perf_counter


def _render(ctx: ProbeContext, messages: List[dict]) -> Any:
    return ctx.make_template("drag", "NQ").get_string(messages=messages)


def generate_returns_text(ctx: ProbeContext) -> Result:
    prompt = _render(ctx, [{"role": "user", "content": "Name one primary colour."}])
    outputs = ctx.generator.generate([prompt])
    ok = isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], str) \
        and bool(outputs[0].strip())
    return ok, f"outputs={str(outputs)[:80]!r}"


def template_renders_messages(ctx: ProbeContext) -> Result:
    text = _render(ctx, [{"role": "system", "content": "ALPHA-SYSTEM-MARK"},
                         {"role": "user", "content": "BETA-USER-MARK"}])
    ok = isinstance(text, str) and "ALPHA-SYSTEM-MARK" in text and "BETA-USER-MARK" in text
    return ok, f"rendered {type(text).__name__} of length {len(text)}"


def batch_order_preserved(ctx: ProbeContext) -> Result:
    prompts = [_render(ctx, [{"role": "user", "content": f"Reply with exactly one word: {word}"}])
               for word in ORDER_WORDS]
    outputs = ctx.generator.generate(prompts, max_tokens=8)
    wrong = [word for word, text in zip(ORDER_WORDS, outputs) if word not in str(text).lower()]
    ok = len(outputs) == len(ORDER_WORDS) and not wrong
    return ok, f"{len(outputs)} outputs; out of order or missing: {wrong}"


def raw_finish_reason(ctx: ProbeContext) -> Result:
    prompt = _render(ctx, [{"role": "user", "content": "Count from one to fifty in words."}])
    raw = ctx.generator.generate([prompt], return_raw_output=True, max_tokens=4)
    reason = getattr(raw[0].outputs[0], "finish_reason", None)
    return reason == "length", f"finish_reason={reason!r} for a 4-token cap"


def batch_search_matches_stored_naive(ctx: ProbeContext) -> Result:
    if not ctx.stored:
        return False, "no stored rows to compare"
    mismatched = []
    for question, ids, _ in ctx.stored:
        got = [doc["id"] for doc in ctx.retriever.batch_search([question])[0]][:len(ids)]
        if got != list(ids):
            mismatched.append(question[:40])
    return not mismatched, f"{len(ctx.stored)} questions; mismatched: {mismatched}"


def _descending(scores: Sequence[Any]) -> bool:
    numbers = [float(s) for s in scores]
    return all(math.isfinite(n) for n in numbers) and numbers == sorted(numbers, reverse=True)


def batch_search_scores(ctx: ProbeContext) -> Result:
    queries = [question for question, _, _ in ctx.stored] or ["who wrote hamlet"]
    rows, scores = ctx.retriever.batch_search(queries, return_score=True)
    ok = len(rows) == len(scores) == len(queries) and all(
        len(r) == len(s) and _descending(s) for r, s in zip(rows, scores))
    return ok, f"{len(rows)} rows, score lengths {[len(s) for s in scores]}"


def index_identity_fast(ctx: ProbeContext) -> Result:
    if not ctx.index_path:
        return False, "no index path"
    start = ctx.clock()
    identity = index_identity(ctx.index_path)
    elapsed = ctx.clock() - start
    return elapsed < 1.0, f"{identity} in {elapsed:.3f} s"


def faiss_gpus(ctx: ProbeContext) -> Result:
    count = ctx.import_module("faiss").get_num_gpus()
    return count == 1, f"faiss.get_num_gpus() = {count}"


def thread_env(ctx: ProbeContext) -> Result:
    want = str(ctx.flags.omp_threads)
    seen = {name: ctx.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    faiss_threads = ctx.import_module("faiss").omp_get_max_threads()
    ok = all(value == want for value in seen.values()) and str(faiss_threads) == want
    return ok, f"want {want}; env {seen}; faiss {faiss_threads}"


def nvidia_smi(ctx: ProbeContext) -> Result:
    if ctx.read_gpu_gib is None:
        return False, "no GPU reader"
    gib = ctx.read_gpu_gib()
    return 70.0 <= gib <= 90.0, f"one card, {gib:.1f} GiB"


def _scored(ctx: ProbeContext, prompts: List[Any], labels: Sequence[str]) -> Tuple[int, int]:
    outputs = ctx.generator.generate_with_scores(prompts, labels=labels)
    margins = [margin({"Proponent": o.label_logprobs.get(labels[0]),
                       "Opponent": o.label_logprobs.get(labels[1])}) for o in outputs]
    return sum(m is not None for m in margins), len(prompts)


def scored_judge_coverage(ctx: ProbeContext) -> Result:
    prompts = [_render(ctx, [{"role": "system", "content": JUDGE_SYSTEM.format(pro=p, opp=o)}])
               for p, o in DEBATES]
    covered, total = _scored(ctx, prompts, LABELS)
    return covered == total, f"{covered}/{total} Judge prompts scored both labels"


def scored_sufficiency_coverage(ctx: ProbeContext) -> Result:
    pools = [(question, {question: docs}) for question, _, docs in ctx.stored] or \
        [("who wrote hamlet", {"who wrote hamlet": [{"id": "1", "contents": "Hamlet\nby Shakespeare"}]})]
    prompts = [_render(ctx, sufficiency_messages(question, pool)) for question, pool in pools]
    covered, total = _scored(ctx, prompts, SUFFICIENCY_LABELS)
    return covered == total, f"{covered}/{total} sufficiency prompts scored both labels"


def scored_finish_reasons(ctx: ProbeContext) -> Result:
    prompt = _render(ctx, [{"role": "system", "content": JUDGE_SYSTEM.format(pro="a", opp="b")}])
    ctx.generator.generate_with_scores([prompt, prompt], labels=LABELS)
    scored = getattr(ctx.generator, "last_finish_reasons", None)
    ctx.generator.generate([prompt])
    after = getattr(ctx.generator, "last_finish_reasons", None)
    ok = scored == [SCORED, SCORED] and after != scored
    return ok, f"scored call {scored}; next text call {after}"


PROBES: Dict[str, Tuple[Callable[[ProbeContext], Result], str]] = {
    "generate_returns_text": (generate_returns_text, "any"),
    "template_renders_messages": (template_renders_messages, "server"),
    "batch_order_preserved": (batch_order_preserved, "server"),
    "raw_finish_reason": (raw_finish_reason, "server"),
    "batch_search_matches_stored_naive": (batch_search_matches_stored_naive, "server"),
    "batch_search_scores": (batch_search_scores, "server"),
    "index_identity_fast": (index_identity_fast, "server"),
    "faiss_gpus": (faiss_gpus, "server"),
    "thread_env": (thread_env, "server"),
    "nvidia_smi": (nvidia_smi, "server"),
    "scored_judge_coverage": (scored_judge_coverage, "any"),
    "scored_sufficiency_coverage": (scored_sufficiency_coverage, "any"),
    "scored_finish_reasons": (scored_finish_reasons, "server"),
}
