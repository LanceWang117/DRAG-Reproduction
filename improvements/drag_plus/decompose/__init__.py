"""Multi-hop decomposition: typed plans, grounded bounded-beam execution, composition.

``decompose`` is off by default. When on, ``route.run_decomposed`` plans at most three
hops, executes them with the J10 anchored store and falls back to the normal pipeline on
any invalid plan, ungrounded binding or exhausted call budget.
"""
from .beam import BEAM_MARGIN, BEAM_WIDTH, Candidate, select_candidates
from .compare import IRCoTPipeline, SelfAskPipeline
from .compose import Composition, CompositionError, LedgerEntry, compose_answer
from .execute import DEFAULT_CALL_CAP, CallBudget, DecompositionResult, execute_plan
from .plan import (MAX_STEPS, Plan, PlanError, SubQuestion, parse_plan, planner_messages,
                   single_hop_plan, validate_plan)
from .route import run_decomposed

__all__ = ["BEAM_MARGIN", "BEAM_WIDTH", "DEFAULT_CALL_CAP", "MAX_STEPS", "CallBudget",
           "Candidate", "Composition", "CompositionError", "DecompositionResult",
           "IRCoTPipeline", "LedgerEntry", "Plan", "PlanError", "SelfAskPipeline",
           "SubQuestion", "compose_answer", "execute_plan", "parse_plan",
           "planner_messages", "run_decomposed", "select_candidates", "single_hop_plan",
           "validate_plan"]
