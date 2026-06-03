"""
ClinIQ — Knowledge Agent
Retrieves relevant clinical guidelines and medical literature from the vector store
using multi-query hybrid retrieval, then synthesises a clinician-grade summary.

Output written to ClinicalState:
  retrieved_knowledge, relevant_guidelines
"""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from core.llm_factory import get_llm
from core.retriever import get_retriever
from core.state import ClinicalState

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a clinical knowledge specialist with access to evidence-based medical guidelines.

Using the retrieved literature provided, synthesise a concise, evidence-based clinical summary tailored to this specific patient's presentation.

Your summary must:
1. Reference specific guidelines by name (e.g. ACC/AHA, ADA, KDIGO)
2. Highlight management priorities for the identified conditions
3. Flag any critical clinical considerations or red-flag findings
4. Note medication considerations relevant to this patient
5. Be written at a senior clinician level (not lay language)

Format: Structured paragraphs by topic. No bullet lists. Max 600 words."""

_HUMAN = """Patient Presentation:
Chief Complaint: {chief_complaint}
Conditions: {conditions}
Medications: {medications}
Key Labs: {lab_values}

Retrieved Clinical Knowledge:
{knowledge}

Synthesise the most clinically relevant guidelines for this patient."""


class KnowledgeAgent:
    """
    Knowledge Agent: Multi-query Hybrid RAG → Guideline Synthesis

    Strategy:
    1. Build multiple retrieval queries from patient data (per-condition + drug combo + labs)
    2. De-duplicate retrieved chunks across queries
    3. Pass top-K to LLM for evidence synthesis
    """

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.1)
        self._retriever = get_retriever()
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM), ("human", _HUMAN)]
        )
        logger.info("KnowledgeAgent initialised")

    # ── Query builder ─────────────────────────────────────────────────────────

    def _build_queries(self, state: ClinicalState) -> List[str]:
        """
        Generate multiple targeted retrieval queries from patient context.
        Multi-query strategy improves recall across diverse conditions.
        """
        queries: List[str] = []

        # Per-condition queries
        for cond in state.get("conditions", [])[:5]:
            queries.append(f"clinical management guidelines {cond}")

        # Drug combination query
        meds = state.get("medications", [])
        if len(meds) >= 2:
            queries.append(f"drug interactions {' '.join(m.split()[0] for m in meds[:4])}")

        # Lab-driven queries
        lab_keys = list(state.get("lab_values", {}).keys())
        if lab_keys:
            queries.append(f"abnormal lab interpretation {' '.join(lab_keys[:4])}")

        # Chief complaint fallback
        if not queries and state.get("chief_complaint"):
            queries.append(state["chief_complaint"])

        # Always include a readmission risk query for inpatient context
        queries.append("30-day readmission risk reduction strategies")

        return queries[:6]  # cap at 6 queries

    def run(self, state: ClinicalState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        trace = state.get("trace_id", "?")
        logger.info(f"KnowledgeAgent.run | trace={trace[:8]}")

        try:
            queries = self._build_queries(state)
            logger.debug(f"KnowledgeAgent: {len(queries)} queries: {queries}")

            # Multi-query retrieval with deduplication
            seen: set = set()
            all_docs: List[Dict] = []

            for query in queries:
                for doc in self._retriever.retrieve(query, k=4):
                    if doc["content"] not in seen:
                        seen.add(doc["content"])
                        all_docs.append(doc)

            # Sort by relevance score, take top 7
            all_docs.sort(key=lambda d: d.get("score", 0), reverse=True)
            top_docs = all_docs[:7]

            # Format knowledge for LLM
            if top_docs:
                knowledge_text = "\n\n---\n\n".join(
                    f"[{doc['metadata'].get('source', 'Medical Literature')}]\n{doc['content']}"
                    for doc in top_docs
                )
            else:
                knowledge_text = (
                    "No specific guidelines retrieved from knowledge base. "
                    "Apply general evidence-based clinical reasoning."
                )

            # LLM synthesis
            chain = self._prompt | self._llm
            resp = chain.invoke({
                "chief_complaint": state.get("chief_complaint", "Not documented"),
                "conditions": ", ".join(state.get("conditions", [])) or "None documented",
                "medications": ", ".join(state.get("medications", [])) or "None documented",
                "lab_values": str(state.get("lab_values", {})) or "Not available",
                "knowledge": knowledge_text,
            })
            summary = resp.content if hasattr(resp, "content") else str(resp)

            elapsed = (time.perf_counter() - t0) * 1000
            step = (
                f"✅ Knowledge Agent ({elapsed:.0f}ms) — "
                f"{len(top_docs)} chunks retrieved across {len(queries)} queries"
            )
            logger.info(step)

            return {
                "retrieved_knowledge": top_docs,
                "relevant_guidelines": [summary],
                "processing_steps": state.get("processing_steps", []) + [step],
            }

        except Exception as exc:
            logger.exception(f"KnowledgeAgent error: {exc}")
            elapsed = (time.perf_counter() - t0) * 1000
            step = f"⚠️ Knowledge Agent failed ({elapsed:.0f}ms): {exc}"
            return {
                "retrieved_knowledge": [],
                "relevant_guidelines": [
                    "Knowledge retrieval unavailable — proceeding with LLM-only assessment."
                ],
                "processing_steps": state.get("processing_steps", []) + [step],
            }
