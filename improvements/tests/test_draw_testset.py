"""Tests for the seeded, stratified test-set sampler."""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
import unittest

IMPROVEMENTS = Path(__file__).resolve().parents[1]
PROJECT = IMPROVEMENTS.parent
sys.path.insert(0, str(PROJECT / "scripts"))
import draw_testset  # noqa: E402

EVIDENCE = PROJECT / "experiments" / "J05-testset"
TEST_SET = EVIDENCE / "questions-test.jsonl"
DEV = PROJECT / "results" / "questions-200.jsonl"
SEALED_SHA256 = "5d76fbe39c36e7b15a3bd4330e6909429f455ff4e885e79a2a0496c2ac338a38"


def independent_normalise(text):
    """Separate formulation of the documented normalisation, for the seal test."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", folded).split())


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def row(dataset, split, qid, question, answers=("gold",)):
    return {"dataset": dataset, "split": split, "id": qid,
            "question": question, "golden_answers": list(answers)}


def write_tree(root):
    """One unique row per configured source plus a dev row that collides with NQ."""
    dev = root / "dev.jsonl"
    write_jsonl(dev, [row("NQ", "dev", "dev_1", "What is Alpha?")])
    for source in draw_testset.SOURCES:
        qid = f"{source.split}_10"
        write_jsonl(root / source.path,
                    [row(source.dataset, source.split, qid, f"Question for {source.dataset}?")])
    return dev


class NormalisationTests(unittest.TestCase):
    def test_normalise_question_folds_case_punctuation_and_space(self):
        self.assertEqual(draw_testset.normalise_question("  Who WROTE \u201cIt\u201d? "),
                         "who wrote it")

    def test_dev_exclusions_keeps_identities_and_normalised_texts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "dev.jsonl"
            write_jsonl(dev, [row("NQ", "dev", "dev_1", "What is Alpha?")])
            keys, texts = draw_testset.dev_exclusions(dev)
        self.assertEqual(keys, {("NQ", "dev", "dev_1")})
        self.assertEqual(texts, {"what is alpha"})


class EligibilityTests(unittest.TestCase):
    def test_excludes_dev_key_dev_text_and_empty_answers(self):
        source = draw_testset.Source("StrategyQA", "strategyqa/train.jsonl", "train")
        rows = [
            row("StrategyQA", "train", "train_7", "A unique question"),
            row("StrategyQA", "train", "train_1", "What is Alpha?"),
            row("StrategyQA", "train", "train_2", "What is Gamma?", answers=()),
            row("StrategyQA", "train", "train_3", "What is Delta?"),
            row("StrategyQA", "train", "train_4", "what is   delta!"),
        ]
        dev_keys = {("StrategyQA", "train", "train_7")}
        chosen = draw_testset.eligible_candidates(rows, source, dev_keys, {"what is alpha"})
        self.assertEqual([item["id"] for item in chosen], ["train_3", "train_4"])
        self.assertEqual(chosen[0]["dataset"], "StrategyQA")
        self.assertEqual(chosen[0]["split"], "train")

    def test_a_row_with_an_empty_question_is_ineligible(self):
        source = draw_testset.Source("NQ", "nq/test.jsonl", "test")
        chosen = draw_testset.eligible_candidates(
            [row("NQ", "test", "test_1", "  ")], source, set(), set())
        self.assertEqual(chosen, [])

    def test_a_row_without_an_id_is_ineligible(self):
        source = draw_testset.Source("NQ", "nq/test.jsonl", "test")
        item = row("NQ", "test", None, "A question")
        self.assertEqual(draw_testset.eligible_candidates([item], source, set(), set()), [])

    def test_an_empty_dev_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "dev.jsonl"
            write_jsonl(dev, [])
            with self.assertRaises(ValueError):
                draw_testset.dev_exclusions(dev)

    def test_candidate_order_does_not_depend_on_source_row_order(self):
        source = draw_testset.Source("NQ", "nq/test.jsonl", "test")
        rows = [row("NQ", "test", "test_2", "Beta?"), row("NQ", "test", "test_1", "Alpha?")]
        first = draw_testset.eligible_candidates(rows, source, set(), set())
        second = draw_testset.eligible_candidates(list(reversed(rows)), source, set(), set())
        self.assertEqual([item["id"] for item in first], ["test_1", "test_2"])
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])


class DrawTests(unittest.TestCase):
    def test_draw_keeps_the_requested_number_in_a_seeded_order(self):
        candidates = [{"dataset": "NQ", "id": f"test_{index}", "question": f"Question {index}?"}
                      for index in range(5)]
        first = draw_testset.draw(candidates, 2, "seed")
        second = draw_testset.draw(list(reversed(candidates)), 2, "seed")
        self.assertEqual(len(first), 2)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])

    def test_shortfall_returns_every_eligible_row(self):
        candidates = [{"dataset": "NQ", "id": "a", "question": "Alpha?"},
                      {"dataset": "NQ", "id": "b", "question": "Beta?"}]
        self.assertEqual(len(draw_testset.draw(candidates, 10, "seed")), 2)

    def test_draw_keeps_one_row_per_normalised_question(self):
        candidates = [
            {"dataset": "NQ", "id": "a", "question": "What is Delta?"},
            {"dataset": "NQ", "id": "b", "question": "what is   delta"},
            {"dataset": "NQ", "id": "c", "question": "What is Gamma?"},
        ]
        chosen = draw_testset.draw(candidates, 10, "seed")
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len({draw_testset.normalise_question(item["question"]) for item in chosen}), 2)

    def test_draw_skips_text_selected_by_an_earlier_dataset(self):
        seen = set()
        first = draw_testset.draw(
            [{"dataset": "NQ", "id": "a", "question": "Same question?"}], 5, "seed", seen)
        second = draw_testset.draw(
            [{"dataset": "HotpotQA", "id": "b", "question": "same   question"}], 5, "seed", seen)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_rank_key_is_pinned(self):
        self.assertEqual(draw_testset.rank_key("20261121", "NQ", "test_0"),
                         "0a0ca671c7568f324b562b05c2eee29332a36aecd52e69e64483091c179cecb9")


class BuildTests(unittest.TestCase):
    def build(self, root, per_dataset=1):
        dev = write_tree(root)
        return draw_testset.build_testset(root, dev, per_dataset, "seed")

    def test_counts_positions_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, report = self.build(Path(tmp))
        self.assertEqual(len(rows), 6)
        self.assertEqual([item["position"] for item in rows], list(range(6)))
        self.assertEqual(sum(entry["drawn"] for entry in report), 6)
        self.assertEqual({entry["dataset"] for entry in report},
                         {source.dataset for source in draw_testset.SOURCES})
        self.assertTrue(all(entry["shortfall"] == 0 for entry in report))

    def test_shortfall_is_reported_when_the_cap_exceeds_eligible_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, report = self.build(Path(tmp), per_dataset=10)
        self.assertEqual(len(rows), 6)
        self.assertEqual({entry["shortfall"] for entry in report}, {9})

    def test_two_builds_render_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = self.build(Path(tmp))
            again, _ = self.build(Path(tmp))
        self.assertEqual(draw_testset.render(rows), draw_testset.render(again))

    def test_missing_source_split_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = write_tree(root)
            (root / "strategyqa" / "train.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                draw_testset.build_testset(root, dev, 1, "seed")


class CommittedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.rows = draw_testset.read_jsonl(TEST_SET)

    def test_committed_set_has_600_unique_rows_under_one_megabyte(self):
        counts = collections.Counter(item["dataset"] for item in self.rows)
        expected = collections.Counter({source.dataset: 100 for source in draw_testset.SOURCES})
        keys = {(item["dataset"], item["split"], item["id"]) for item in self.rows}
        texts = {draw_testset.normalise_question(item["question"]) for item in self.rows}
        self.assertEqual(counts, expected)
        self.assertEqual(len(self.rows), 600)
        self.assertEqual((len(keys), len(texts)), (600, 600))
        self.assertEqual([item["position"] for item in self.rows], list(range(600)))
        self.assertLessEqual(TEST_SET.stat().st_size, 1024 * 1024)
        for item in self.rows:
            self.assertTrue(draw_testset.has_reference_answer(item))
            self.assertEqual(set(item), {"dataset", "split", "id", "question",
                                         "golden_answers", "position"})

    def test_committed_set_has_zero_key_and_text_overlap_with_dev(self):
        dev = draw_testset.read_jsonl(DEV)
        dev_keys = {(item["dataset"], item["split"], str(item["id"])) for item in dev}
        dev_texts = {independent_normalise(item["question"]) for item in dev}
        keys = {(item["dataset"], item["split"], item["id"]) for item in self.rows}
        texts = {independent_normalise(item["question"]) for item in self.rows}
        self.assertEqual(keys & dev_keys, set())
        self.assertEqual(texts & dev_texts, set())

    def test_sealed_file_hash_is_pinned(self):
        self.assertEqual(hashlib.sha256(TEST_SET.read_bytes()).hexdigest(), SEALED_SHA256)

    def test_checksums_cover_and_match_the_sealed_files(self):
        entries = {}
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split()
                entries[name] = digest
        self.assertEqual(set(entries),
                         {"README.md", "PREREGISTRATION.md", "questions-test.jsonl"})
        for name, digest in entries.items():
            self.assertEqual(hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                             digest, name)


if __name__ == "__main__":
    unittest.main()
