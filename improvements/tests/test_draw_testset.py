"""Tests for the seeded, stratified test-set sampler."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

IMPROVEMENTS = Path(__file__).resolve().parents[1]
PROJECT = IMPROVEMENTS.parent
sys.path.insert(0, str(PROJECT / "scripts"))
import draw_testset  # noqa: E402


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
    def test_excludes_dev_key_dev_text_empty_answers_and_duplicates(self):
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
        self.assertEqual([item["id"] for item in chosen], ["train_3"])
        self.assertEqual(chosen[0]["dataset"], "StrategyQA")
        self.assertEqual(chosen[0]["split"], "train")

    def test_a_row_with_an_empty_question_is_ineligible(self):
        source = draw_testset.Source("NQ", "nq/test.jsonl", "test")
        chosen = draw_testset.eligible_candidates(
            [row("NQ", "test", "test_1", "  ")], source, set(), set())
        self.assertEqual(chosen, [])


class DrawTests(unittest.TestCase):
    def test_draw_keeps_the_requested_number_in_a_seeded_order(self):
        candidates = [{"dataset": "NQ", "id": f"test_{index}"} for index in range(5)]
        first = draw_testset.draw(candidates, 2, "seed")
        second = draw_testset.draw(list(reversed(candidates)), 2, "seed")
        self.assertEqual(len(first), 2)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])

    def test_shortfall_returns_every_eligible_row(self):
        candidates = [{"dataset": "NQ", "id": "a"}, {"dataset": "NQ", "id": "b"}]
        self.assertEqual(len(draw_testset.draw(candidates, 10, "seed")), 2)


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


if __name__ == "__main__":
    unittest.main()
