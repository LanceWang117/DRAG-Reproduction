"""J07: duplicate-query guard, zero-novelty exit and the recorded fixture cases."""
import importlib.util
import json
from pathlib import Path
import unittest

from drag_plus.flags import Flags
from drag_plus.pipeline import DragPlusPipeline
from drag_plus.query_norm import is_duplicate_query, normalize_query
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from t0_scenarios import query_round

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "zero_novelty.json"
QUESTION = "Who was the screenwriter of the 1986 Telugu film Tandra Paparayudu?"
DUPLICATE = "Who is the screenwriter of the 1986 Telugu film Tandra Paparayudu?"
DIFFERENT = "Who is the screenwriter of the 1986 Telugu film Sankarabharanam?"
ANSWER = ["final answer"]


def passages(*ids):
    return [{"id": pid, "contents": f"passage {pid}"} for pid in ids]


def run(question, outputs, docs, **flag_values):
    gen = ScriptedGenerator(outputs)
    retriever = FakeRetriever(docs)
    values = {"max_answer_debate_rounds": 0, **flag_values}
    pipeline = DragPlusPipeline(Flags(values), gen, retriever, IdentityTemplate(), "NQ")
    state = pipeline.run_item(RecordingItem(question))
    return state, gen, retriever


def exit_decisions(state):
    return [row for row in state.decisions if row["decision"] == "query_exit"]


class NormalisationTests(unittest.TestCase):
    def test_auxiliary_verbs_and_articles_do_not_make_a_new_query(self):
        self.assertEqual(normalize_query(QUESTION), normalize_query(DUPLICATE))
        self.assertTrue(is_duplicate_query(DUPLICATE, [QUESTION]))
        self.assertTrue(is_duplicate_query("The answer", ["a answer!"]))

    def test_a_different_entity_is_not_a_duplicate(self):
        self.assertNotEqual(normalize_query(QUESTION), normalize_query(DIFFERENT))
        self.assertFalse(is_duplicate_query(DIFFERENT, [QUESTION]))

    def test_nfc_case_punctuation_and_whitespace_are_normalised(self):
        composed = "Café—au lait!"
        decomposed = "  cafe\u0301 au   lait  "
        self.assertEqual(normalize_query(composed), normalize_query(decomposed))
        self.assertEqual(normalize_query(composed), frozenset({"café", "au", "lait"}))

    def test_stop_only_and_empty_queries_have_the_empty_key(self):
        self.assertEqual(normalize_query("Is was the of for"), frozenset())
        self.assertEqual(normalize_query(""), frozenset())
        self.assertTrue(is_duplicate_query("was", ["is"]))


class DuplicateGuardTests(unittest.TestCase):
    DOCS = {QUESTION: passages("q1"), DUPLICATE: passages("d1")}

    def test_duplicate_expansion_skips_the_search_and_ends_the_stage(self):
        outputs = query_round("Opponent", f"weak. Query Expansion: {DUPLICATE}") + ANSWER
        state, _, retriever = run(QUESTION, outputs, self.DOCS, max_query_debate_rounds=3,
                                  skip_duplicate_query=True)
        self.assertEqual(retriever.calls, [QUESTION])
        self.assertEqual(list(state.query_pool), [QUESTION])
        self.assertEqual(exit_decisions(state),
                         [{"stage": "query", "round": 0, "decision": "query_exit",
                           "status": "duplicate"}])
        self.assertEqual(state.stage, "answer")

    def test_flag_off_keeps_the_baseline_search_and_append(self):
        outputs = (query_round("Opponent", f"weak. Query Expansion: {DUPLICATE}")
                   + query_round("Proponent") + ANSWER)
        state, _, retriever = run(QUESTION, outputs, self.DOCS, max_query_debate_rounds=2)
        self.assertEqual(retriever.calls, [QUESTION, DUPLICATE])
        self.assertEqual(list(state.query_pool), [QUESTION, DUPLICATE])
        self.assertEqual(exit_decisions(state), [])

    def test_optimization_onto_another_key_is_a_duplicate(self):
        docs = {QUESTION: passages("q1"), "second query": passages("s1")}
        outputs = (query_round("Opponent", "weak. Query Expansion: second query")
                   + query_round("Opponent", f"weak. Query Optimization: second query -> {QUESTION}")
                   + ANSWER)
        state, _, retriever = run(QUESTION, outputs, docs, max_query_debate_rounds=3,
                                  skip_duplicate_query=True)
        self.assertEqual(retriever.calls, [QUESTION, "second query"])
        self.assertEqual(list(state.query_pool), [QUESTION, "second query"])
        self.assertEqual(exit_decisions(state)[0]["status"], "duplicate")


class ZeroNoveltyTests(unittest.TestCase):
    def test_no_new_id_ends_the_stage_without_appending(self):
        docs = {QUESTION: passages("a", "b"), "second query": passages("a")}
        outputs = query_round("Opponent", "weak. Query Expansion: second query") + ANSWER
        state, _, retriever = run(QUESTION, outputs, docs, max_query_debate_rounds=3,
                                  zero_novelty_exit=True)
        self.assertEqual(retriever.calls, [QUESTION, "second query"])
        self.assertEqual(list(state.query_pool), [QUESTION])
        self.assertEqual(exit_decisions(state),
                         [{"stage": "query", "round": 0, "decision": "query_exit",
                           "status": "zero_novelty"}])

    def test_novel_ids_never_trigger_an_exit(self):
        docs = {QUESTION: passages("a"), "second query": passages("b")}
        outputs = (query_round("Opponent", "weak. Query Expansion: second query")
                   + query_round("Proponent") + ANSWER)
        state, _, _ = run(QUESTION, outputs, docs, max_query_debate_rounds=2,
                          zero_novelty_exit=True)
        self.assertEqual(list(state.query_pool), [QUESTION, "second query"])
        self.assertEqual(exit_decisions(state), [])

    def test_optimization_restores_the_pool_it_would_have_deleted(self):
        docs = {QUESTION: passages("a", "b"), "who wrote the novel": passages("a")}
        outputs = (query_round("Opponent",
                               "weak. Query Optimization: who wrote it -> who wrote the novel")
                   + ANSWER)
        state, _, retriever = run(QUESTION, outputs, docs, max_query_debate_rounds=3,
                                  zero_novelty_exit=True)
        self.assertEqual(retriever.calls, [QUESTION, "who wrote the novel"])
        self.assertEqual(list(state.query_pool), [QUESTION])
        self.assertEqual(state.query_pool[QUESTION], passages("a", "b"))

    def test_round_counts_and_batching_keep_the_exit(self):
        docs = {QUESTION: passages("a"), "second query": passages("a")}
        for rounds in (1, 2, 3):
            for batched in (False, True):
                with self.subTest(rounds=rounds, batched=batched):
                    outputs = (query_round("Opponent", "weak. Query Expansion: second query")
                               + ANSWER)
                    state, _, _ = run(QUESTION, outputs, docs,
                                      max_query_debate_rounds=rounds,
                                      batch_agents_in_round=batched,
                                      zero_novelty_exit=True)
                    self.assertEqual(exit_decisions(state)[0]["round"], 0)
                    self.assertEqual(list(state.query_pool), [QUESTION])


class RecordedFixtureTests(unittest.TestCase):
    @staticmethod
    def load():
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    @staticmethod
    def action(case, query):
        if case["kind"] == "single_replacement":
            return f"weak. Query Optimization: {case['steps'][0]['query']} -> {query}"
        return f"weak. Query Expansion: {query}"

    def replay(self, case):
        steps = case["steps"]
        docs = {step["query"]: passages(*step["ids"]) for step in steps}
        outputs = []
        for step in steps[1:case["exit_step"] + 1]:
            outputs += query_round("Opponent", self.action(case, step["query"]))
        state, _, retriever = run(steps[0]["query"], outputs + ANSWER, docs,
                                  max_query_debate_rounds=len(steps),
                                  zero_novelty_exit=True)
        return case, state, retriever

    def test_each_recorded_case_exits_at_the_recorded_step(self):
        for case in self.load()["cases"]:
            with self.subTest(case=case["case"]):
                case, state, retriever = self.replay(case)
                self.assertEqual(exit_decisions(state),
                                 [{"stage": "query", "round": case["exit_step"] - 1,
                                   "decision": "query_exit", "status": "zero_novelty"}])
                if case["kind"] == "multi":
                    expected = [s["query"] for s in case["steps"][:case["exit_step"]]]
                else:
                    expected = [case["steps"][0]["query"]]
                self.assertEqual(list(state.query_pool), expected)
                self.assertEqual(len(retriever.calls), case["exit_step"] + 1)

    def test_fixture_matches_its_builder(self):
        path = FIXTURE.with_name("build_zero_novelty.py")
        spec = importlib.util.spec_from_file_location("build_zero_novelty", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(self.load(), module.build())


if __name__ == "__main__":
    unittest.main()
