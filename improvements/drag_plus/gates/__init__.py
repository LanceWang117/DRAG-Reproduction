"""Escalation gate: label-free features, a fitted risk model and lazy sufficiency (J14).

Nothing here imports FlashRAG, vLLM, torch or FAISS; the model is plain JSON and the
scored call is injected through ``Backends.complete_with_scores``.
"""
from .escalation import (GateDecision, GateModel, GateModelError, SufficiencyDecision,
                         calibrated_risk, decide, load_model, resolve_sufficiency,
                         sufficiency_messages, sufficiency_probability)
from .features import FEATURE_NAMES, compute_features

__all__ = ["FEATURE_NAMES", "GateDecision", "GateModel", "GateModelError",
           "SufficiencyDecision", "calibrated_risk", "compute_features", "decide",
           "load_model", "resolve_sufficiency", "sufficiency_messages",
           "sufficiency_probability"]
