"""ClinicalMind StateGraph — LangGraph workflow (state-driven, no interrupt()).

Graph topology:

    START
      │
      ▼
  [entry]  ← safety check + intent classification
      │
      ▼
  route_after_entry:
   ├─ diagnosis → [interview_generate] ──→ END (pause)
   │                 ↑                      │
   │                 │                      ▼ (console collects answers,
   │                 │                 resumes with human_response in state)
   │                 │                      │
   │                 │                [interview_process]
   │                 │                      │
   │                 │                route_after_process:
   │                 │                 ├─ interviewing → interview_generate
   │                 └─────────────────┘
   │                 │
   │                 │ (phase=diagnosing)
   │                 ▼
   │              [diagnose]
   │                 │
   │                 ▼
   │              [treatment_plan] → END
   │
   ├─ research → [research] → END
   └─ planning → [standalone_planning] → END

Human-in-the-loop is achieved by pausing between graph runs:
- interview_generate sets questions + phase="awaiting_human", graph ends
- Console reads questions, gets user input, fills human_response
- Console runs graph again → interview_process → route → (loop or diagnose)
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import ClinicalState
from .nodes import (
    entry_node,
    interview_generate_node,
    interview_process_node,
    diagnose_node,
    treatment_plan_node,
    research_node,
    standalone_planning_node,
    route_after_entry,
    route_after_process,
    route_after_diagnosis,
)


def build_graph() -> StateGraph:
    """Build and compile the ClinicalMind LangGraph workflow.

    Uses state-driven pauses instead of interrupt() for LangGraph 1.x compatibility.
    MemorySaver persists state across graph invocations for the interview loop.
    """

    workflow = StateGraph(ClinicalState)

    # ── Nodes ──────────────────────────────────────────────────────────
    workflow.add_node("entry", entry_node)
    workflow.add_node("interview_generate", interview_generate_node)
    workflow.add_node("interview_process", interview_process_node)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("treatment_plan", treatment_plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("standalone_planning", standalone_planning_node)

    # ── Edges ──────────────────────────────────────────────────────────
    workflow.set_entry_point("entry")

    # Entry routing
    workflow.add_conditional_edges(
        "entry",
        route_after_entry,
        {
            "interview_generate": "interview_generate",
            "interview_process": "interview_process",
            "research": "research",
            "standalone_planning": "standalone_planning",
            "diagnose": "diagnose",
        },
    )

    # Interview generate → END (pause; console picks up questions)
    workflow.add_edge("interview_generate", END)

    # Interview process → loop back or go to diagnose
    workflow.add_conditional_edges(
        "interview_process",
        route_after_process,
        {
            "interview_generate": "interview_generate",
            "diagnose": "diagnose",
        },
    )

    # Diagnose → treatment plan or end
    workflow.add_conditional_edges(
        "diagnose",
        route_after_diagnosis,
        {"treatment_plan": "treatment_plan", "__end__": END},
    )

    # Terminal
    workflow.add_edge("treatment_plan", END)
    workflow.add_edge("research", END)
    workflow.add_edge("standalone_planning", END)

    # ── Compile ────────────────────────────────────────────────────────
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


graph = build_graph()
