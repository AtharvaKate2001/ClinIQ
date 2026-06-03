"""
ClinIQ — Report Agent
Final agent in the pipeline. Synthesises all upstream outputs into a structured,
professional clinical intelligence report suitable for clinician review.

Output written to ClinicalState:
  final_report  (executive_summary, clinical_findings, recommendations, follow_up, metadata)
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from core.llm_factory import get_llm
from core.state import ClinicalState

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a senior clinician generating a structured clinical intelligence report for a complex patient.

Return ONLY valid JSON (no markdown) with this structure:
{{
  "executive_summary": "2-3 sentence high-priority clinical overview for a clinician reading quickly",
  "clinical_findings": [
    {{
      "category": "Category name (e.g. Cardiac, Metabolic, Renal)",
      "finding": "Specific clinical finding",
      "significance": "normal|abnormal|critical",
      "recommendation": "Specific clinical action"
    }}
  ],
  "recommendations": [
    "Specific, actionable, prioritised recommendation 1",
    "Specific, actionable, prioritised recommendation 2"
  ],
  "follow_up_actions": [
    "Immediate action (within hours/days)",
    "Short-term follow-up (within weeks)"
  ],
  "clinical_narrative": "2-3 paragraph clinical summary suitable for a medical record or handoff note. Use clinical language. Incorporate risk findings and guideline-based management.",
  "discharge_considerations": ["consideration 1", "consideration 2"]
}}

Requirements:
- Be specific, not generic (e.g. 'Hold metformin given eGFR 22' not 'Review medications')
- Prioritise by clinical urgency (most critical first)
- Reference specific scores/values from the data
- clinical_findings: minimum 4, maximum 10"""

_HUMAN = """Generate a comprehensive clinical intelligence report from this full patient analysis:

PATIENT: {patient_summary}
CHIEF COMPLAINT: {chief_complaint}
CONDITIONS: {conditions}
MEDICATIONS: {medications}
LAB VALUES: {lab_values}

RISK ASSESSMENT:
{risk_summary}

DRUG INTERACTIONS / RULE ALERTS:
{drug_interactions}

CLINICAL GUIDELINES SUMMARY:
{guidelines}

PIPELINE AUDIT:
{processing_steps}"""


class ReportAgent:
    """
    Report Agent: Multi-input synthesis → Structured Clinical Report

    This is the output agent — it reads everything upstream agents wrote to state
    and produces a unified, structured report for clinical decision support.
    """

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.2)  # slight creativity for narrative prose
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM), ("human", _HUMAN)]
        )
        logger.info("ReportAgent initialised")

    @staticmethod
    def _clean_json(raw: str) -> str:
        s = raw.strip()
        if s.startswith("```"):
            inner = s.split("```")[1]
            s = inner[4:].strip() if inner.startswith("json") else inner.strip()
        return s

    @staticmethod
    def _format_risk_summary(risk_scores: Dict, risk_level: str) -> str:
        lines = [f"Overall Risk Level: {risk_level}", ""]
        if not risk_scores:
            return lines[0]
        for cat, data in risk_scores.items():
            if not isinstance(data, dict):
                continue
            score = data.get("score", 0)
            level = data.get("level", "?")
            factors = data.get("factors", [])
            justification = data.get("justification", "")
            lines.append(f"• {cat.replace('_', ' ').title()}: {level} ({score:.0%})")
            if justification:
                lines.append(f"  Reasoning: {justification}")
            if factors:
                lines.append(f"  Factors: {', '.join(factors[:4])}")
        return "\n".join(lines)

    @staticmethod
    def _format_drug_interactions(interactions: List, rule_alerts: List) -> str:
        parts: List[str] = []
        for di in interactions:
            parts.append(
                f"• {di.get('drug_1','?')} + {di.get('drug_2','?')} "
                f"[{di.get('severity','?').upper()}]: {di.get('description','')}. "
                f"Recommendation: {di.get('recommendation','')}"
            )
        for alert in rule_alerts:
            parts.append(f"• RULE ALERT: {alert}")
        return "\n".join(parts) if parts else "None detected"

    def run(self, state: ClinicalState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        trace = state.get("trace_id", "?")
        logger.info(f"ReportAgent.run | trace={trace[:8]}")

        try:
            pi = state.get("patient_info", {})
            patient_summary = (
                f"Name: {pi.get('name', 'Anonymous')} | "
                f"Age: {pi.get('age', 'Unknown')} | "
                f"Gender: {pi.get('gender', 'Unknown')} | "
                f"Allergies: {', '.join(pi.get('allergies', ['NKDA']))}"
            )

            chain = self._prompt | self._llm
            resp = chain.invoke({
                "patient_summary": patient_summary,
                "chief_complaint": state.get("chief_complaint", "Not documented"),
                "conditions": ", ".join(state.get("conditions", [])) or "None documented",
                "medications": ", ".join(state.get("medications", [])) or "None documented",
                "lab_values": json.dumps(state.get("lab_values", {}), indent=2) or "{}",
                "risk_summary": self._format_risk_summary(
                    state.get("risk_scores", {}), state.get("risk_level", "UNKNOWN")
                ),
                "drug_interactions": self._format_drug_interactions(
                    state.get("drug_interactions", []),
                    state.get("contraindications", []),
                ),
                "guidelines": (state.get("relevant_guidelines", ["Not available"])[0])[:1500],
                "processing_steps": "\n".join(state.get("processing_steps", [])),
            })

            raw = resp.content if hasattr(resp, "content") else str(resp)
            report_data: Dict = json.loads(self._clean_json(raw))

            elapsed = (time.perf_counter() - t0) * 1000
            step = f"✅ Report Agent ({elapsed:.0f}ms) — Clinical report generated"
            logger.info(step)

            return {
                "final_report": {
                    **report_data,
                    "metadata": {
                        "trace_id": state.get("trace_id"),
                        "generated_at": datetime.utcnow().isoformat() + "Z",
                        "pipeline_steps": state.get("processing_steps", []) + [step],
                        "agents_run": ["intake_agent", "knowledge_retrieval", "risk_assessment", "report_generation"],
                        "domain": "healthcare",
                    },
                },
                "processing_steps": state.get("processing_steps", []) + [step],
            }

        except Exception as exc:
            logger.exception(f"ReportAgent error: {exc}")
            elapsed = (time.perf_counter() - t0) * 1000
            step = f"⚠️ Report Agent error ({elapsed:.0f}ms): {exc}"
            return {
                "final_report": {
                    "executive_summary": f"Report generation encountered an error: {exc}. "
                                         "Manual clinical review required.",
                    "clinical_findings": [],
                    "recommendations": ["Immediate manual clinical review — automated report failed"],
                    "follow_up_actions": ["Contact on-call clinician"],
                    "clinical_narrative": state.get("chief_complaint", "Patient presented for evaluation."),
                    "discharge_considerations": [],
                    "metadata": {
                        "trace_id": state.get("trace_id"),
                        "error": str(exc),
                        "generated_at": datetime.utcnow().isoformat() + "Z",
                    },
                },
                "processing_steps": state.get("processing_steps", []) + [step],
            }
