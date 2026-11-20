"""Checkpoint records and the logs derived from them.

One checkpoint record is the durable truth for one (arm, question). The transcript and
timing logs are derived from the records in schedule order, so a resumed run writes the
same logs as an uninterrupted one. ``t_start`` is rewritten to seconds since the
question's first call: absolute clocks do not survive a restart.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Tuple

from ..answers import native_pred, unified_pred
from ..instrument.timing import LOG_FIELDS


class WorkItem:
    """The minimal dataset item the pipelines need: question, id, ``update_output``."""

    def __init__(self, row: Dict[str, Any]):
        self.question = row["question"]
        self.id = row["id"]
        self.golden_answers = row["golden_answers"]
        self.output: Dict[str, Any] = {}

    def update_output(self, key: str, value: Any) -> None:
        self.output[key] = value


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_ids(docs: Iterable[dict]) -> List[Any]:
    return [doc.get("id") for doc in docs]


def _compact(key: str, value: Any) -> Any:
    """Keep passage ids, not passage text, for the big retrieval outputs."""
    if key == "QueryStage_QueryPool":
        return {query: _doc_ids(docs) for query, docs in value.items()}
    if key == "retrieval_result":
        return _doc_ids(value)
    return value


def transcript_entries(outputs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
    return [{"key": key, "sha256": _digest(value), "value": _compact(key, value)}
            for key, value in outputs]


def relative_calls(calls: List[dict]) -> List[dict]:
    if not calls:
        return []
    origin = calls[0]["t_start"]
    return [dict(row, t_start=round(row["t_start"] - origin, 6)) for row in calls]


def build_record(order: int, arm: str, row: Dict[str, Any], state: Any) -> Dict[str, Any]:
    raw = state.raw_pred if state.raw_pred is not None else ""
    return {"order": order, "arm": arm, "dataset": row["dataset"], "split": row["split"],
            "id": row["id"], "question": row["question"],
            "golden_answers": row["golden_answers"], "pred": state.pred, "raw_pred": raw,
            "pred_native": native_pred(arm, raw),
            "pred_unified": unified_pred(raw, row["dataset"]),
            "entries": transcript_entries(state.outputs),
            "calls": relative_calls(state.calls)}


def derive_logs(records: List[Dict[str, Any]], run_id: str, variant: str
                ) -> Tuple[List[dict], List[dict]]:
    """Return (timing rows, transcript rows) for the records, ordered by schedule position."""
    timings: List[dict] = []
    transcripts: List[dict] = []
    for record in sorted(records, key=lambda r: r["order"]):
        who = {"run_id": run_id, "variant": variant, "arm": record["arm"],
               "dataset": record["dataset"], "split": record["split"], "id": record["id"]}
        timings += [{**who, **call} for call in record["calls"]]
        transcripts.append({**who, "entries": record["entries"], "pred": record["pred"],
                            "pred_native": record["pred_native"],
                            "pred_unified": record["pred_unified"]})
    return timings, transcripts


def row_has_full_schema(row: Dict[str, Any]) -> bool:
    return all(field in row for field in LOG_FIELDS)
