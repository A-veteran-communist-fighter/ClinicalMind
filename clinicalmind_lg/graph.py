"""ClinicalMind StateGraph — LangGraph workflow with interrupt() for human-in-the-loop.

Graph topology (simplified):

    START
      │
      ▼
  [entry]  ← safety check + intent classification
      │
      ▼
  route_after_entry:
   ├─ diagnosis → [interview_loop] ←──┐
   │                │                  │
   │                ▼ (interrupt)      │
   │            [wait for human]       │ resume via Command(resume=...)
   │                │                  │
   │                └──────────────────┘ (loop: ask → answer → ask → ...)
   │                │
   │                ▼ (synthesis triggered)
   │            [diagnose]
   │                │
   │                ▼
   │            [treatment_plan] → END
   │
   ├─ research → [research] → END
   └─ planning → [standalone_planning] → END
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import ClinicalState
from .nodes import (
    entry_node,
    interview_loop_node,
    diagnose_node,
    treatment_plan_node,
    research_node,
    standalone_planning_node,
    route_after_entry,
    route_after_diagnosis,
)


def build_graph() -> StateGraph:
    """Build and compile the ClinicalMind LangGraph workflow.

    Returns a compiled graph with:
    - Memory checkpointing (required by interrupt())
    - interrupt() boundaries for human-in-the-loop
    """

    workflow = StateGraph(ClinicalState)

    # ── Add nodes ──────────────────────────────────────────────────────
    workflow.add_node("entry", entry_node)
    workflow.add_node("interview_loop", interview_loop_node)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("treatment_plan", treatment_plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("standalone_planning", standalone_planning_node)

    # ── Edges ──────────────────────────────────────────────────────────
    workflow.set_entry_point("entry")

    # Entry routes to the appropriate pipeline
    workflow.add_conditional_edges(
        "entry",
        route_after_entry,
        {
            "interview_loop": "interview_loop",
            "research": "research",
            "standalone_planning": "standalone_planning",
            "diagnose": "diagnose",  # emergency shortcut
        },
    )

    # Interview loop → diagnose (loop exits when info sufficient)
    workflow.add_edge("interview_loop", "diagnose")

    # Diagnose → treatment plan or end (emergency skips plan)
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
    # MemorySaver is REQUIRED for interrupt() to work
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Module-level compiled graph instance
graph = build_graph()
