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
LONG_QUERY = ("which british biscuit maker won a 1991 vat tribunal arguing that its orange "
              "sponge snack is legally a cake")  # > 64 characters: B2 truncated these
LLAMA3_SYSTEM = "<|start_header_id|>system<|end_header_id|>"
LLAMA3_GENERATION = "<|start_header_id|>assistant<|end_header_id|>\n\n"
MIB_PER_GIB = 1024.0


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
    read_gpu_rows: Optional[Callable[[], List[float]]] = None  # GiB per visible card
    clock: Callable[[], float] = time.perf_counter


def _render(ctx: ProbeContext, messages: List[dict]) -> Any:
    return ctx.make_template("drag", "NQ").get_string(messages=messages)


def generate_returns_text(ctx: ProbeContext) -> Result:
    prompt = _render(ctx, [{"role": "user", "content": "Name one primary colour."}])
    outputs = ctx.generator.generate([prompt])
    ok = isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], str) \
        and bool(outputs[0].strip())
    return ok, f"outputs={str(outputs)[:80]!r}"


def _llama3_ok(text: Any, *marks: str) -> bool:
    """System header, the marks in order, and the assistant generation header last."""
    if not isinstance(text, str) or LLAMA3_SYSTEM not in text or not text.endswith(LLAMA3_GENERATION):
        return False
    positions = [text.find(mark) for mark in marks]
    return all(p >= 0 for p in positions) and positions == sorted(positions)


def template_renders_messages(ctx: ProbeContext) -> Result:
    """The Llama-3.1 chat template with a generation prompt, for both prompt shapes."""
    both = _render(ctx, [{"role": "system", "content": "ALPHA-SYSTEM-MARK"},
                         {"role": "user", "content": "BETA-USER-MARK"}])
    system_only = _render(ctx, [{"role": "system", "content": "GAMMA-JUDGE-MARK"}])
    ok = _llama3_ok(both, "ALPHA-SYSTEM-MARK", "BETA-USER-MARK") and \
        _llama3_ok(system_only, "GAMMA-JUDGE-MARK")
    return ok, f"system+user ends {str(both)[-60:]!r}; system-only ends {str(system_only)[-60:]!r}"


def _ids(docs: Any) -> List[str]:
    return [doc["id"] for doc in docs]


def b2_search_vs_batch(ctx: ProbeContext) -> Result:
    """B2 on the real index: raw search(q) vs batch_search([q])[0].

    A short query must agree on both paths. For a query over 64 characters the detail
    records whether raw ``search`` diverges (B2's root cause predicts it does; without
    ``retrieval_cache`` the DRAG query stage uses that raw path).
    """
    short = "who wrote hamlet"
    agree = _ids(ctx.retriever.search(short)) == _ids(ctx.retriever.batch_search([short])[0])
    long_single = _ids(ctx.retriever.search(LONG_QUERY))
    long_batch = _ids(ctx.retriever.batch_search([LONG_QUERY])[0])
    verdict = "B2 reproduced: long query diverges" if long_single != long_batch else \
        "B2 not reproduced: long query agrees"
    return agree and bool(long_batch), f"short agrees={agree}; {verdict} ({long_single[:3]} vs {long_batch[:3]})"


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
    has_stat = identity.count(":") >= 2  # '<path>:<size>:<mtime_ns>'; a bare path means stat failed
    return elapsed < 1.0 and has_stat, f"{identity} in {elapsed:.3f} s"


def faiss_gpus(ctx: ProbeContext) -> Result:
    count = ctx.import_module("faiss").get_num_gpus()
    return count == 1, f"faiss.get_num_gpus() = {count}"


def thread_env(ctx: ProbeContext) -> Result:
    """What FAISS (OpenMP) and torch (intra-op, MKL) actually use, not what env says."""
    want = ctx.flags.omp_threads
    faiss_threads = ctx.import_module("faiss").omp_get_max_threads()
    torch_threads = ctx.import_module("torch").get_num_threads()
    env = {name: ctx.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    ok = faiss_threads == want and torch_threads == want
    return ok, f"want {want}; faiss {faiss_threads}; torch {torch_threads}; env {env}"


def parse_gpu_rows(text: str) -> List[float]:
    """``nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`` -> GiB per card."""
    return [float(line) / MIB_PER_GIB for line in text.splitlines() if line.strip()]


def nvidia_smi(ctx: ProbeContext) -> Result:
    if ctx.read_gpu_rows is None:
        return False, "no GPU reader"
    rows = ctx.read_gpu_rows()
    ok = len(rows) == 1 and 70.0 <= rows[0] <= 90.0
    return ok, f"{len(rows)} visible card(s): {[round(r, 1) for r in rows]} GiB"


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
    "b2_search_vs_batch": (b2_search_vs_batch, "server"),
    "batch_search_scores": (batch_search_scores, "server"),
    "index_identity_fast": (index_identity_fast, "server"),
    "faiss_gpus": (faiss_gpus, "server"),
    "thread_env": (thread_env, "server"),
    "nvidia_smi": (nvidia_smi, "server"),
    "scored_judge_coverage": (scored_judge_coverage, "any"),
    "scored_sufficiency_coverage": (scored_sufficiency_coverage, "any"),
    "scored_finish_reasons": (scored_finish_reasons, "server"),
}
