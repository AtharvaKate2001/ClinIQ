"""
ClinIQ — LangGraph Agent State
TypedDict state shared across all agents in the pipeline.
Each agent reads from and writes to this shared state object.
"""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class ClinicalState(TypedDict):
    """
    Shared state flowing through the LangGraph multi-agent pipeline.

    Flow: intake → knowledge_retrieval → risk_assessment → report_generation
    Each agent appends to `processing_steps` and may overwrite its own fields.

    The `messages` field uses `operator.add` so each agent CAN append messages
    rather than replacing them — standard LangGraph pattern for conversation memory.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    raw_document: str
    document_type: str            # medical_report | lab_results | discharge_summary | prescription

    # ── Intake Agent Output ───────────────────────────────────────────────────
    patient_info: Dict[str, Any]  # name, age, gender, blood_type, allergies, vitals
    conditions: List[str]         # diagnosed conditions / ICD codes
    medications: List[str]        # current medications
    lab_values: Dict[str, Any]    # key: value pairs (e.g. "HbA1c": "9.2%")
    chief_complaint: str

    # ── Knowledge Agent Output ────────────────────────────────────────────────
    retrieved_knowledge: List[Dict[str, Any]]  # [{content, metadata, score}]
    relevant_guidelines: List[str]             # synthesized guideline summaries

    # ── Risk Agent Output ─────────────────────────────────────────────────────
    risk_scores: Dict[str, Any]   # {category: {score, level, factors, justification}}
    risk_factors: List[str]       # top priority clinical concerns
    risk_level: str               # overall: LOW | MODERATE | HIGH | CRITICAL
    drug_interactions: List[Dict[str, Any]]  # [{drug_1, drug_2, severity, description, recommendation}]
    contraindications: List[str]

    # ── Report Agent Output ───────────────────────────────────────────────────
    final_report: Dict[str, Any]  # executive_summary, findings, recommendations, metadata

    # ── Pipeline Metadata ─────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    processing_steps: List[str]   # human-readable audit trail
    trace_id: str
