"""
ClinIQ — Intake Agent
First agent in the pipeline. Parses the raw clinical document and extracts
structured patient data using LLM + Pydantic-validated JSON output.

Output written to ClinicalState:
  patient_info, conditions, medications, lab_values, chief_complaint, document_type
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from core.llm_factory import get_fast_llm
from core.state import ClinicalState

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a specialist medical document parser. Extract structured information from clinical documents with high precision.

Return ONLY valid JSON (no markdown, no preamble) with this exact structure:
{{
  "patient_info": {{
    "name": "string or null",
    "age": integer_or_null,
    "gender": "Male|Female|Other|null",
    "blood_type": "string or null",
    "height": "string with unit or null",
    "weight": "string with unit or null",
    "allergies": ["allergy_1", "allergy_2"]
  }},
  "chief_complaint": "primary reason for visit or admission as a concise phrase",
  "conditions": ["Condition 1", "Condition 2"],
  "medications": ["Drug 1 dose frequency", "Drug 2 dose frequency"],
  "lab_values": {{
    "test_name": "value with unit"
  }},
  "vital_signs": {{
    "blood_pressure": "string or null",
    "heart_rate": "string or null",
    "temperature": "string or null",
    "oxygen_saturation": "string or null",
    "respiratory_rate": "string or null"
  }},
  "document_type": "medical_report|lab_results|discharge_summary|prescription|clinical_note"
}}

Rules:
- Extract only explicitly mentioned information (no inference)
- Use null for missing fields (never omit keys)
- Include ALL medications with doses if available
- Capture ALL lab results mentioned"""

_HUMAN = """Parse this clinical document:

{document}

Return the JSON object only."""

# ── Agent ─────────────────────────────────────────────────────────────────────

class IntakeAgent:
    """
    Intake Agent: Document → Structured Patient Data

    Uses a fast LLM (smaller model) for deterministic extraction.
    Falls back to empty structures on parse failure to keep the pipeline alive.
    """

    def __init__(self) -> None:
        self._llm = get_fast_llm(temperature=0.0)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM), ("human", _HUMAN)]
        )
        logger.info("IntakeAgent initialised")

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences if the model wrapped its JSON."""
        s = raw.strip()
        if s.startswith("```"):
            parts = s.split("```")
            # parts[1] may be "json\n{...}" or just "{..."
            inner = parts[1] if len(parts) > 1 else s
            s = inner[4:].strip() if inner.startswith("json") else inner.strip()
        return s

    def run(self, state: ClinicalState) -> Dict[str, Any]:
        """Execute intake extraction and return state update dict."""
        t0 = time.perf_counter()
        trace = state.get("trace_id", "?")
        logger.info(f"IntakeAgent.run | trace={trace[:8]}")

        # Defensive: truncate very long documents to avoid token overflow
        doc = state["raw_document"][:8000]

        try:
            chain = self._prompt | self._llm
            resp = chain.invoke({"document": doc})
            raw = resp.content if hasattr(resp, "content") else str(resp)
            extracted: Dict = json.loads(self._clean_json(raw))

            patient_info = extracted.get("patient_info", {})
            conditions = extracted.get("conditions", [])
            medications = extracted.get("medications", [])
            lab_values = extracted.get("lab_values", {})

            # Merge vital_signs into patient_info for easy downstream access
            if extracted.get("vital_signs"):
                patient_info["vital_signs"] = extracted["vital_signs"]

            elapsed = (time.perf_counter() - t0) * 1000
            step = (
                f"✅ Intake Agent ({elapsed:.0f}ms) — "
                f"{len(conditions)} condition(s), "
                f"{len(medications)} medication(s), "
                f"{len(lab_values)} lab value(s)"
            )
            logger.info(step)

            return {
                "patient_info": patient_info,
                "conditions": conditions,
                "medications": medications,
                "lab_values": lab_values,
                "chief_complaint": extracted.get("chief_complaint", ""),
                "document_type": extracted.get("document_type", "medical_report"),
                "processing_steps": state.get("processing_steps", []) + [step],
                "error": None,
            }

        except json.JSONDecodeError as exc:
            logger.error(f"IntakeAgent JSON parse error: {exc}")
            return self._fallback(state, f"JSON parse error: {exc}", t0)

        except Exception as exc:
            logger.exception(f"IntakeAgent unexpected error: {exc}")
            return self._fallback(state, str(exc), t0)

    def _fallback(self, state: ClinicalState, err_msg: str, t0: float) -> Dict[str, Any]:
        elapsed = (time.perf_counter() - t0) * 1000
        step = f"⚠️ Intake Agent failed ({elapsed:.0f}ms): {err_msg}"
        return {
            "patient_info": {},
            "conditions": [],
            "medications": [],
            "lab_values": {},
            "chief_complaint": "Extraction failed — see raw document",
            "document_type": "unknown",
            "processing_steps": state.get("processing_steps", []) + [step],
            "error": err_msg,
        }
