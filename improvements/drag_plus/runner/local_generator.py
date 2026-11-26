"""HTTP generator for the local smoke backend: Ollama's OpenAI-compatible chat API.

Standard library only. It sends message lists to ``/v1/chat/completions`` (never a chat
string built with a template), retries a timeout, a dropped connection or an HTTP 5xx at
most ``retries`` times, and returns outputs in input order. Ollama truncates an over-long
prompt without saying so, and the OpenAI endpoint cannot set the context size, so the
adapter fits every prompt to the context limit itself and counts what it trimmed.

``generate_with_scores`` (J24) asks the same endpoint for the first positions' top
log-probabilities and maps label prefixes to scores (``label_scores``), so the J12
log-prob Judge and the J14 sufficiency scorer run off the A100. ``last_finish_reasons``
holds the server's real ``finish_reason`` per prompt of the latest call, for the
per-call timing rows.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..judge.logprob import SCORED, ScoredOutput
from .label_scores import scored_labels

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_URL = "http://localhost:11434/v1"
CONTEXT_TOKENS = 4096
MAX_TOKENS = 256
SEED = 2024
RETRIES = 2
SCORE_TOKENS = 12  # the label plus leading markup, so the text fallback still parses
TOP_LOGPROBS = 20
TIMEOUT_SECONDS = 180.0
CHARS_PER_TOKEN = 3  # deliberately low, so the estimate errs on the side of trimming
TRIM_MARK = "\n[... trimmed to fit the local context ...]\n"
_FORWARDED = {"max_tokens": "max_tokens", "max_new_tokens": "max_tokens",
              "temperature": "temperature", "seed": "seed", "stop": "stop"}


class LocalGeneratorError(RuntimeError):
    """The local server could not produce a completion."""


class ModelNotFoundError(LocalGeneratorError):
    """HTTP 404: the server does not have the model."""


class _Retryable(Exception):
    """A failure worth another attempt: timeout, dropped connection, HTTP 5xx."""


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def _cut_middle(text: str, excess: int) -> str:
    """Remove up to ``excess`` characters from the middle; never longer than the input."""
    keep = max(len(text) - excess - len(TRIM_MARK), 0)
    head = keep * 2 // 3
    tail = text[len(text) - (keep - head):] if keep > head else ""
    return text[:head] + TRIM_MARK + tail


def fit_messages(messages: List[dict], budget_tokens: int) -> Tuple[List[dict], bool]:
    """Cut the middle out of the longest messages until the estimate fits the budget."""
    excess = sum(len(m["content"]) for m in messages) - budget_tokens * CHARS_PER_TOKEN
    fitted = list(messages)
    for index in sorted(range(len(fitted)), key=lambda i: -len(fitted[i]["content"])):
        text = fitted[index]["content"]
        if excess <= 0 or len(text) <= len(TRIM_MARK):
            break
        fitted[index] = dict(fitted[index], content=_cut_middle(text, excess))
        excess -= len(text) - len(fitted[index]["content"])
    return fitted, fitted != list(messages)


def _is_conversation(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(m, dict) for m in value)


def as_conversations(prompts: Any) -> List[List[dict]]:
    """One prompt (text or message list) or a batch of them, as message lists."""
    if isinstance(prompts, str):
        return [[{"role": "user", "content": prompts}]]
    if _is_conversation(prompts):
        return [prompts]
    return [as_conversations(prompt)[0] for prompt in prompts]


@dataclass
class GeneratorStats:
    calls: int = 0
    retries: int = 0
    empty_outputs: int = 0
    trimmed_prompts: int = 0
    clamped_max_tokens: int = 0
    scored_calls: int = 0
    finish_reasons: Dict[str, int] = field(default_factory=dict)
    ignored_params: Set[str] = field(default_factory=set)

    def as_dict(self) -> Dict[str, Any]:
        return {"calls": self.calls, "retries": self.retries, "empty_outputs": self.empty_outputs,
                "trimmed_prompts": self.trimmed_prompts,
                "clamped_max_tokens": self.clamped_max_tokens,
                "scored_calls": self.scored_calls,
                "finish_reasons": dict(self.finish_reasons),
                "ignored_params": sorted(self.ignored_params)}


class OllamaGenerator:
    """Satisfies the pipeline's ``Generator`` protocol: ``generate(prompts, **params)``."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL, *,
                 timeout: float = TIMEOUT_SECONDS, retries: int = RETRIES,
                 max_tokens: int = MAX_TOKENS, seed: int = SEED,
                 context_tokens: int = CONTEXT_TOKENS,
                 opener: Callable = urllib.request.urlopen,
                 sleep: Callable[[float], None] = time.sleep):
        self.model, self.base_url, self.seed = model, base_url.rstrip("/"), seed
        self.stats = GeneratorStats()
        self._timeout, self._retries, self._max_tokens = timeout, retries, max_tokens
        self._context, self._opener, self._sleep = context_tokens, opener, sleep
        self.last_finish_reasons: Optional[List[str]] = None

    def generate(self, prompts: Any, **params: Any) -> List[str]:
        options = self._options(params)
        self.last_finish_reasons = []
        return [self._text(self._choice(messages, options))
                for messages in as_conversations(prompts)]

    def generate_with_scores(self, prompts: Any, labels: Sequence[str],
                             **params: Any) -> List[ScoredOutput]:
        """One scored completion per prompt: the text plus one log-prob per label."""
        options = self._options({"max_tokens": SCORE_TOKENS, **params})
        options.update(logprobs=True, top_logprobs=TOP_LOGPROBS)
        self.last_finish_reasons = []
        outputs = []
        for messages in as_conversations(prompts):
            choice = self._choice(messages, options)
            text = choice["message"]["content"] or ""
            self.stats.scored_calls += 1
            self.last_finish_reasons.append(SCORED)
            outputs.append(ScoredOutput(text, scored_labels(choice, text, labels)))
        return outputs

    def _options(self, params: Dict[str, Any]) -> Dict[str, Any]:
        options = {"temperature": 0.0, "max_tokens": self._max_tokens, "seed": self.seed}
        for name, value in params.items():
            if name in _FORWARDED:
                options[_FORWARDED[name]] = value
            else:
                self.stats.ignored_params.add(name)
        if options["max_tokens"] > self._context // 2:
            options["max_tokens"] = self._context // 2  # leave at least half for the prompt
            self.stats.clamped_max_tokens += 1
        return options

    def _choice(self, messages: List[dict], options: Dict[str, Any]) -> Dict[str, Any]:
        fitted, trimmed = fit_messages(messages, self._context - options["max_tokens"])
        self.stats.trimmed_prompts += int(trimmed)
        payload = {"model": self.model, "messages": fitted, "stream": False, **options}
        body = self._request(json.dumps(payload).encode("utf-8"))
        self.stats.calls += 1
        try:
            choice = body["choices"][0]
            choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LocalGeneratorError(f"unexpected response shape: {str(body)[:200]}") from error
        return choice

    def _request(self, data: bytes) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._post(data)
            except _Retryable as error:
                if attempt >= self._retries:
                    raise LocalGeneratorError(
                        f"{self.base_url} failed after {attempt + 1} attempts: {error}") from error
                attempt += 1
                self.stats.retries += 1
                self._sleep(0.5 * attempt)

    def _post(self, data: bytes) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with self._opener(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise self._http_error(error) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as error:
            raise _Retryable(str(error)) from error
        except ValueError as error:
            raise LocalGeneratorError(f"the server sent a body that is not JSON: {error}") from error

    def _http_error(self, error: urllib.error.HTTPError) -> Exception:
        detail = error.read().decode("utf-8", "replace")[:200]
        if error.code == 404:
            return ModelNotFoundError(
                f"HTTP 404 from {self.base_url}: model {self.model!r} not found (run "
                f"`ollama pull {self.model}`) or the base URL is wrong")
        if error.code >= 500:
            return _Retryable(f"HTTP {error.code}: {detail}")
        return LocalGeneratorError(f"HTTP {error.code} from {self.base_url}: {detail}")

    def _text(self, choice: Dict[str, Any]) -> str:
        text = choice["message"]["content"]
        reason = choice.get("finish_reason") or "unknown"
        self.stats.finish_reasons[reason] = self.stats.finish_reasons.get(reason, 0) + 1
        if self.last_finish_reasons is not None:
            self.last_finish_reasons.append(reason)
        if not text or not text.strip():
            self.stats.empty_outputs += 1
            return ""
        return text
