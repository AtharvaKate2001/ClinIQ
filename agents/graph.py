"""
ClinIQ — LangGraph Multi-Agent Workflow
Defines the stateful agent graph:

  START
    ↓
  [intake_agent]          → Parse document, extract structured patient data
    ↓
  [knowledge_retrieval]   → Hybrid RAG over medical knowledge base
    ↓
  [risk_assessment]       → Multi-dimensional clinical risk scoring + drug checks
    ↓
  [report_generation]     → Synthesise everything into structured clinical report
    ↓
  END

LangGraph handles: state passing, conditional edges, checkpointing (resumability),
and makes it trivial to add parallel branches or human-in-the-loop nodes.
"""
from __future__ import annotations
import logging
import os
from typing import Any

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.state import ClinicalState

logger = logging.getLogger(__name__)


# ── Conditional routing functions ─────────────────────────────────────────────

def _route_after_intake(state: ClinicalState) -> str:
    """Skip to report if intake agent hit a fatal error."""
    if state.get("error"):
        logger.warning(f"Intake error detected — routing directly to report: {state['error']}")
        return "report_generation"
    return "knowledge_retrieval"


# ── Graph factory ─────────────────────────────────────────────────────────────

def create_workflow() -> Any:
    """
    Build and compile the ClinIQ LangGraph workflow.

    Returns a compiled CompiledStateGraph with MemorySaver checkpointing.
    Each run can be resumed by passing the same thread_id in config.
    """
    # Import agents here to avoid circular imports
    from agents.intake_agent import IntakeAgent
    from agents.knowledge_agent import KnowledgeAgent
    from agents.risk_agent import RiskAgent
    from agents.report_agent import ReportAgent

    intake_agent = IntakeAgent()
    knowledge_agent = KnowledgeAgent()
    risk_agent = RiskAgent()
    report_agent = ReportAgent()

    # ── Build graph ───────────────────────────────────────────────────────────
    workflow = StateGraph(ClinicalState)

    # Register nodes — each node is a callable(state) → state
    workflow.add_node("intake_agent", intake_agent.run)
    workflow.add_node("knowledge_retrieval", knowledge_agent.run)
    workflow.add_node("risk_assessment", risk_agent.run)
    workflow.add_node("report_generation", report_agent.run)

    # Entry point
    workflow.add_edge(START, "intake_agent")

    # After intake: either proceed or skip to report on error
    workflow.add_conditional_edges(
        "intake_agent",
        _route_after_intake,
        {
            "knowledge_retrieval": "knowledge_retrieval",
            "report_generation": "report_generation",
        },
    )

    # Linear sequential flow
    workflow.add_edge("knowledge_retrieval", "risk_assessment")
    workflow.add_edge("risk_assessment", "report_generation")
    workflow.add_edge("report_generation", END)

    # ── Checkpointer (in-memory; swap to SqliteSaver for persistence) ─────────
    # SqliteSaver example (persistent across restarts):
    #   from langgraph.checkpoint.sqlite import SqliteSaver
    #   memory = SqliteSaver.from_conn_string("./data/checkpoints/workflow.db")
    memory = MemorySaver()

    compiled = workflow.compile(checkpointer=memory)
    logger.info("✅ LangGraph workflow compiled")
    return compiled


# ── Singleton ─────────────────────────────────────────────────────────────────
_workflow: Any = None


def get_workflow() -> Any:
    """Return module-level singleton compiled workflow."""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow
