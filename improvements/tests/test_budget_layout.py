"""J11: budget selection, evidence-first prefixes and prefix-caching wiring."""
import random
from collections import Counter
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus import prompts  # noqa: E402
from drag_plus.evidence.budget import BudgetError, select_entries  # noqa: E402
from drag_plus.evidence.format import EvidenceEntry, format_entries  # noqa: E402
from drag_plus.evidence.layout import round_prefix, shared_prefix  # noqa: E402
from drag_plus.evidence.store import AnchoredEvidenceStore, EvidencePool  # noqa: E402
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.instrument.timing import whitespace_tokens  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from drag_plus.runner.backends import build_generator  # noqa: E402

from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402


def entry(doc_id, contents, rank=1, qid=1, anchored=False, round_index=0):
    return EvidenceEntry(doc_id, contents, f"E{qid}-{rank}", qid, rank, round_index,
                         anchored)


def words_of(entries, delimiters=False):
    return whitespace_tokens(format_entries(entries, delimiters))


def build_pool():
    store = AnchoredEvidenceStore()
    store.add_query("question", [{"id": "d1", "contents": "Alpha\nbody one"}], 0,
                    anchored=True)
    store.add_query("refined", [{"id": "d2", "contents": "Beta\nbody two"}], 0)
    return EvidencePool(store.to_query_pool(), entries=store.entries)


class BudgetSelectionTests(unittest.TestCase):
    def test_budget_is_never_exceeded_and_anchored_passages_survive(self):
        rng = random.Random(7)
        for case in range(40):
            entries = [entry(f"d{i}", f"Article {i % 4}\n" + " ".join(
                f"w{(i * 7 + j) % 11}" for j in range(4 + i % 6)),
                rank=i % 5 + 1, qid=i // 5 + 1, anchored=i < 3) for i in range(12)]
            fixed = "header question history"
            anchored = [item for item in entries if item.anchored]
            floor = whitespace_tokens(fixed) + words_of(anchored)
            budget = floor + rng.randint(0, 40)
            selection = select_entries(entries, whitespace_tokens, budget, fixed)
            labels = {item.label for item in selection.entries}
            self.assertLessEqual(selection.context_tokens, budget, case)
            self.assertLessEqual(whitespace_tokens(fixed)
                                 + words_of(selection.entries), budget, case)
            for item in anchored:
                self.assertIn(item.label, labels, case)
            by_article = Counter(item.contents.split("\n")[0].casefold()
                                 for item in selection.entries if not item.anchored)
            self.assertLessEqual(max(by_article.values(), default=0), 2, case)

    def test_anchored_material_is_truncated_with_a_warning_not_dropped(self):
        anchored = [entry("a", "Long\n" + " ".join(f"token{i}" for i in range(40)),
                          rank=1, anchored=True),
                    entry("b", "Short\nsmall", rank=1, qid=2, anchored=True)]
        budget = whitespace_tokens("Q\n") + words_of(anchored) - 15
        selection = select_entries(anchored, whitespace_tokens, budget, "Q\n")
        self.assertTrue(selection.truncated)
        self.assertTrue(selection.warnings)
        self.assertEqual({item.label for item in selection.entries}, {"E1-1", "E2-1"})
        self.assertLessEqual(selection.context_tokens, budget)
        self.assertLess(len(selection.entries[0].contents),
                        len(anchored[0].contents))
        self.assertEqual(selection.entries[0].contents.split("\n")[0], "Long")

    def test_a_budget_below_the_anchored_minimum_is_an_error(self):
        anchored = [entry("a", "Long\n" + " ".join(f"t{i}" for i in range(30)),
                          rank=1, anchored=True)]
        with self.assertRaises(BudgetError):
            select_entries(anchored, whitespace_tokens, 2, "Q\n")

    def test_per_article_cap_keeps_two_chunks(self):
        entries = [entry(f"d{i}", "Same article\n" + f"body {i}", rank=i + 1)
                   for i in range(6)]
        selection = select_entries(entries, whitespace_tokens, 1000)
        self.assertEqual(len(selection.entries), 2)
        self.assertEqual(selection.dropped, tuple(f"E1-{i}" for i in range(3, 7)))

    def test_budgeted_evidence_renders_the_same_text_it_counted(self):
        entries = [entry(f"d{i}", f"T{i}\nbody {i} " + "x " * (i * 3), rank=i + 1)
                   for i in range(5)]
        selection = select_entries(entries, whitespace_tokens, 25, "Q")
        rendered = format_entries(selection.entries)
        self.assertEqual(whitespace_tokens("Q") + whitespace_tokens(rendered),
                         selection.context_tokens)
        self.assertLessEqual(selection.context_tokens, 25)


class EvidenceFirstLayoutTests(unittest.TestCase):
    def test_three_roles_share_one_prefix_that_ends_after_the_evidence(self):
        pool = build_pool()
        prefix, selection = round_prefix("who?", pool, None, whitespace_tokens)
        self.assertIsNone(selection)
        messages = [
            prompts.query_agent_system("Proponent Agent 0", prefix=prefix),
            prompts.query_agent_system("Opponent Agent 0", prefix=prefix),
            prompts.query_moderator_system({"Proponent Agent 0": "a",
                                            "Opponent Agent 0": "b"}, "who?", pool,
                                           prefix=prefix)]
        contents = [message["content"] for message in messages]
        for content in contents:
            self.assertTrue(content.startswith(prefix))
        self.assertTrue(prefix.endswith("body two\n"))
        self.assertEqual(prompts.query_round_user("who?", pool, prefix=prefix)["content"],
                         prompts.PROCEED)

    def test_round_r_prefix_is_a_prefix_of_round_r_plus_one(self):
        store = AnchoredEvidenceStore()
        store.add_query("q0", [{"id": "d1", "contents": "A\none"}], 0, anchored=True)
        pool0 = EvidencePool(store.to_query_pool(), entries=store.entries)
        prefix0, _ = round_prefix("who?", pool0, None, whitespace_tokens)
        store.add_query("q1", [{"id": "d2", "contents": "B\ntwo"}], 1)
        pool1 = EvidencePool(store.to_query_pool(), entries=store.entries)
        prefix1, _ = round_prefix("who?", pool1, None, whitespace_tokens)
        self.assertTrue(prefix1.startswith(prefix0))
        self.assertEqual(prefix1, shared_prefix(
            "who?", "E1-1(Title: A) one\nE2-1(Title: B) two"))

    def test_answer_roles_share_the_same_prefix(self):
        pool = build_pool()
        prefix, _ = round_prefix("who?", pool, None, whitespace_tokens)
        messages = [
            prompts.answer_agent_system("Proponent Agent 0", pool, False, prefix=prefix),
            prompts.answer_agent_system("Opponent Agent 0", pool, False, prefix=prefix),
            prompts.answer_moderator_system(False, prefix=prefix)]
        self.assertTrue(all(m["content"].startswith(prefix) for m in messages))
        self.assertTrue(prefix.endswith("body two\n"))


class EvidenceFirstPipelineTests(unittest.TestCase):
    def run_pipeline(self, flags, outputs, docs):
        generator = ScriptedGenerator(outputs)
        item = RecordingItem("who wrote it?")
        pipeline = DragPlusPipeline(flags, generator, FakeRetriever(docs),
                                    IdentityTemplate(), "NQ")
        return item, pipeline.run([item])[0]

    def test_query_and_answer_prompts_start_with_the_evidence_prefix(self):
        flags = Flags({"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
                       "prompt_layout": "evidence_first", "evidence_store": "anchored"})
        outputs = ["proponent argument", "no action here", "Proponent",
                   "proponent answer", "opponent answer", "The answer is: final"]
        docs = {"who wrote it?": [{"id": "d1", "contents": "Alpha\nbody one"}]}
        item, state = self.run_pipeline(flags, outputs, docs)
        pool = state.query_pool
        for stage in ("QueryStage", "AnswerStage"):
            texts = [item.output[f"{stage}_Proponent Agent 0_Round0_InputPrompt"][0]["content"],
                     item.output[f"{stage}_Opponent Agent 0_Round0_InputPrompt"][0]["content"]]
            moderator = item.output[f"{stage}_Moderator_Round0_InputPrompt"]
            texts.append(moderator[0]["content"])
            expected, _ = round_prefix("who wrote it?", pool, None, whitespace_tokens)
            for text in texts:
                self.assertTrue(text.startswith(expected))
            self.assertTrue(expected.endswith("body one\n"))

    def test_role_first_budget_renders_the_selected_evidence(self):
        flags = Flags({"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
                       "evidence_store": "anchored", "context_token_budget": 50})
        outputs = ["proponent argument", "no action here", "Proponent",
                   "proponent answer", "opponent answer", "The answer is: final"]
        docs = {"who wrote it?": [{"id": f"d{i}", "contents": f"Article {i}\n"
                                    + " ".join(f"word{i}_{j}" for j in range(4))}
                                  for i in range(3)]}
        item, _state = self.run_pipeline(flags, outputs, docs)
        prompt = item.output["QueryStage_Proponent Agent 0_Round0_InputPrompt"]
        self.assertLessEqual(whitespace_tokens(prompt[1]["content"]), 50)
        for i in range(3):
            self.assertIn(f"E1-{i + 1}", prompt[1]["content"])

    def test_budgeted_evidence_first_stays_under_the_budget(self):
        flags = Flags({"max_query_debate_rounds": 1, "max_answer_debate_rounds": 1,
                       "prompt_layout": "evidence_first", "evidence_store": "anchored",
                       "context_token_budget": 75})
        outputs = ["proponent argument", "no action here", "Proponent",
                   "proponent answer", "opponent answer", "The answer is: final"]
        docs = {"who wrote it?": [{"id": f"d{i}", "contents": f"Article {i}\n"
                                    + " ".join(f"word{i}_{j}" for j in range(8))}
                                  for i in range(4)]}
        _item, state = self.run_pipeline(flags, outputs, docs)
        prefix, selection = round_prefix("who wrote it?", state.query_pool, 75,
                                         whitespace_tokens)
        self.assertIsNotNone(selection)
        self.assertLessEqual(selection.context_tokens, 75)
        self.assertLessEqual(whitespace_tokens(prefix), 75)


class FakeBase:
    def __init__(self, config):
        self._config = config
        self.model_path = config["model_path"]
        self.tensor_parallel_size = 1
        self.gpu_memory_utilization = 0.85
        self.max_model_len = 4096
        self.use_lora = False


class FakeVLLM(FakeBase):
    pass


class PrefixCachingTests(unittest.TestCase):
    def test_flag_on_passes_enable_prefix_caching_to_the_llm_constructor(self):
        calls = []

        def llm_factory(path, **kwargs):
            calls.append((path, kwargs))
            return "llm"

        generator = build_generator(
            {"model_path": "/model"}, Flags({"enable_prefix_caching": True}),
            lambda config: "flashrag", FakeVLLM, FakeBase,
            llm_factory=llm_factory, tokenizer_factory=lambda path, **kw: "tokenizer")
        self.assertIsInstance(generator, FakeVLLM)
        self.assertEqual(generator.model, "llm")
        self.assertEqual(calls[0][0], "/model")
        self.assertTrue(calls[0][1]["enable_prefix_caching"])
        self.assertEqual(generator.tokenizer, "tokenizer")

    def test_flag_off_uses_flashrags_unchanged_factory(self):
        calls = []

        def llm_factory(path, **kwargs):
            calls.append(path)
            return "llm"

        sentinel = object()
        generator = build_generator({"model_path": "/model"}, Flags(),
                                    lambda config: sentinel, FakeVLLM, FakeBase,
                                    llm_factory=llm_factory)
        self.assertIs(generator, sentinel)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
