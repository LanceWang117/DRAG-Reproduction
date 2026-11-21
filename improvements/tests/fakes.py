"""Scripted, recording stand-ins for the injected generator, retriever and template."""
import copy

from drag_plus.adapters import prompt_count


class ScriptedGenerator:
    """Return the next scripted output per prompt and record every call.

    A batched call consumes one scripted output per prompt, in input order, like the
    real generator contract.
    """

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompts, **params):
        self.calls.append((copy.deepcopy(prompts), dict(params)))
        count = prompt_count(prompts)
        if len(self.outputs) < count:
            raise AssertionError(f"scripted outputs exhausted at call {len(self.calls)}")
        return [self.outputs.pop(0) for _ in range(count)]


class FakeRetriever:
    """Map a query to docs; unknown queries get one deterministic document."""

    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        if query in self.docs:
            return copy.deepcopy(self.docs[query])
        return [{"id": f"doc::{query}", "contents": f"Title for {query}\nBody about {query}"}]

    def batch_search(self, queries):
        return [self.search(query) for query in queries]


class IdentityTemplate:
    """Return the messages unchanged, as the frozen Judge tests do.

    A snapshot is returned because the real template builds a new string on every call;
    returning the live list would let later appends rewrite recorded prompts.
    """

    def __init__(self):
        self.calls = []

    def get_string(self, **kwargs):
        self.calls.append(tuple(sorted(kwargs)))
        return copy.deepcopy(kwargs["messages"])


class RecordingItem:
    """A dataset item that records ``update_output`` calls like FlashRAG's Item."""

    def __init__(self, question, item_id="q0"):
        self.question = question
        self.id = item_id
        self.output = {}
        self.calls = []

    def update_output(self, key, value):
        self.calls.append((key, copy.deepcopy(value)))
        self.output[key] = value

    @property
    def pred(self):
        return self.output["pred"]
