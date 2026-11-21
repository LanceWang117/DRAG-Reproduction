"""Shared helpers for the J06 flag tests: run the frozen baseline and drag_plus, compare.

The two flags are answer-preserving, so every test states what must be *identical* to
the frozen v1.0.1 pipeline and what is allowed to differ (call grouping, absent keys).
"""
import frozen
from drag_plus.adapters import prompt_count
from drag_plus.flags import Flags
from drag_plus.pipeline import DragPlusPipeline
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from t0_scenarios import query_round

QUESTION = "who wrote it?"
MODERATOR_MARKER = "You are a moderator in a debate competition"


class FrozenBaseline(frozen.DebateAugmentedRAG):
    def evaluate(self, dataset, do_eval=True):
        return dataset


def script(query_rounds=2, answer_rounds=2, final_only=False):
    """The scripted outputs for a run: query rounds that end, then answer rounds.

    A final-only script omits the moderator outputs of rounds 0..R-2, because those
    calls are never made (the baseline's script would leave them unconsumed).
    """
    outputs = query_round("Proponent") if query_rounds == 1 else (
        query_round("Opponent", "weak. Query Expansion: second query")
        + query_round("Proponent"))
    for r in range(answer_rounds):
        outputs += [f"proponent answer {r}", f"opponent answer {r}"]
        if not final_only or r == answer_rounds - 1:
            outputs.append(f"The answer is: moderator {r}")
    return outputs


def run_baseline(outputs, query_rounds=2, answer_rounds=2):
    gen, ret, template = ScriptedGenerator(outputs), FakeRetriever(), IdentityTemplate()
    pipeline = FrozenBaseline(
        {"dataset_name": "NQ"}, prompt_template=template,
        max_query_debate_rounds=query_rounds, max_answer_debate_rounds=answer_rounds,
        generator=gen, retriever=ret)
    item = RecordingItem(QUESTION)
    pipeline.run([item])
    return item, gen


def run_plus(outputs, values=None, query_rounds=2, answer_rounds=2, recorder=None,
             generator=None):
    gen = generator or ScriptedGenerator(outputs)
    flags = {"max_query_debate_rounds": query_rounds,
             "max_answer_debate_rounds": answer_rounds, **(values or {})}
    pipeline = DragPlusPipeline(Flags(flags), gen, FakeRetriever(), IdentityTemplate(),
                                "NQ", recorder=recorder)
    item = RecordingItem(QUESTION)
    return item, gen, pipeline.run_item(item)


def prompts_of(calls):
    """Every recorded call flattened to one prompt per prompt, in order."""
    flattened = []
    for prompts, _params in calls:
        flattened.extend(prompts if prompt_count(prompts) > 1 else [prompts])
    return flattened


def is_moderator_prompt(prompt):
    return isinstance(prompt, list) and any(
        MODERATOR_MARKER in message.get("content", "") for message in prompt)


def drop_early_moderator_prompts(prompts, rounds):
    """Baseline prompts minus the moderator prompts of rounds 0..R-2, in order."""
    kept, seen = [], 0
    for prompt in prompts:
        if is_moderator_prompt(prompt):
            seen += 1
            if seen < rounds:
                continue
        kept.append(prompt)
    return kept


def is_early_moderator(key, rounds):
    if not key.startswith("AnswerStage_Moderator_Round"):
        return False
    return int(key.split("Round")[1].split("_")[0]) < rounds - 1


def drop_early_moderator_outputs(calls, rounds):
    """The baseline transcript minus the early moderator outputs and their pred writes.

    ``pred`` is written by every moderator, so only the last write remains.
    """
    total_preds = sum(1 for key, _ in calls if key == "pred")
    kept, seen = [], 0
    for key, value in calls:
        if key == "pred" and seen < total_preds - 1:
            seen += 1
            continue
        if not is_early_moderator(key, rounds):
            kept.append((key, value))
    return kept
