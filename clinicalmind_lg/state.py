"""ClinicalMind LangGraph State Schema.

Defines the shared state that flows through all agent nodes.
Uses LangGraph's TypedDict + Annotated reducers for clean state management.
"""

from typing import Annotated, Any, TypedDict
from operator import add


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: merge two dicts, with b overwriting a on key conflicts."""
    return {**a, **b}


def _append_list(a: list, b: list) -> list:
    """Reducer: concatenate two lists (for additive accumulation)."""
    return a + b


class ClinicalState(TypedDict):
    """Central state object flowing through the ClinicalMind agent graph.

    All nodes read from and write to this state. LangGraph handles
    checkpointing and state persistence automatically.
    """

    # ── Messages (LangChain message history) ──
    messages: Annotated[list, _append_list]

    # ── User input ──
    chief_complaint: str
    current_user_input: str

    # ── Interview state ──
    collected_info: Annotated[dict[str, str], _merge_dicts]
    raw_answers: Annotated[dict[str, str], _merge_dicts]
    asked_questions: Annotated[list[str], _append_list]
    current_questions: list[dict[str, Any]]  # questions to show user now
    interview_round: int

    # ── Clinical reasoning ──
    differential_diagnoses: list[dict[str, Any]]
    red_flags: Annotated[list[str], _append_list]

    # ── Safety triage ──
    triage_flags: list[str]
    is_emergency: bool

    # ── Intent / Routing ──
    intent: str
    intent_confidence: str
    route: str  # "diagnosis" | "research" | "planning" | "monitoring"

    # ── Outputs ──
    diagnosis_report: dict[str, Any]
    treatment_plan: dict[str, Any]
    research_answer: str

    # ── Flow control ──
    phase: str  # "classify" | "safety_check" | "interviewing" | "diagnosing" | "planning" | "research" | "completed"
    action: str  # "ask" | "synthesize" | "end"

    # ── Human-in-the-loop ──
    needs_human_input: bool
    human_response: str


def initial_state(chief_complaint: str) -> ClinicalState:
    """Factory: create a fresh state for a new consultation."""
    return ClinicalState(
        messages=[],
        chief_complaint=chief_complaint,
        current_user_input=chief_complaint,
        collected_info={},
        raw_answers={},
        asked_questions=[],
        current_questions=[],
        interview_round=0,
        differential_diagnoses=[],
        red_flags=[],
        triage_flags=[],
        is_emergency=False,
        intent="",
        intent_confidence="",
        route="diagnosis",
        diagnosis_report={},
        treatment_plan={},
        research_answer="",
        phase="classify",
        action="ask",
        needs_human_input=False,
        human_response="",
    )
