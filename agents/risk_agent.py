"""
ClinIQ — Risk Agent
Computes multi-dimensional clinical risk scores using a two-layer approach:
  Layer 1: Deterministic rule-based checks (anticoagulant combos, critical labs)
  Layer 2: LLM-based reasoning for nuanced risk scoring

Also identifies drug-drug interactions from the medication list.

Output written to ClinicalState:
  risk_scores, risk_level, risk_factors, drug_interactions, contraindications
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from core.llm_factory import get_llm
from core.state import ClinicalState

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a clinical risk stratification specialist. Your task is to calculate evidence-based risk scores for a patient based on their clinical presentation.

Return ONLY valid JSON (no markdown) with this structure:
{{
  "risk_scores": {{
    "30_day_readmission": {{
      "score": 0.0,
      "level": "LOW|MODERATE|HIGH|CRITICAL",
      "factors": ["factor 1", "factor 2"],
      "justification": "Brief clinical reasoning (1-2 sentences)"
    }},
    "cardiovascular": {{
      "score": 0.0,
      "level": "LOW|MODERATE|HIGH|CRITICAL",
      "factors": [],
      "justification": ""
    }},
    "acute_kidney_injury": {{
      "score": 0.0,
      "level": "LOW|MODERATE|HIGH|CRITICAL",
      "factors": [],
      "justification": ""
    }},
    "medication_adverse_event": {{
      "score": 0.0,
      "level": "LOW|MODERATE|HIGH|CRITICAL",
      "factors": [],
      "justification": ""
    }}
  }},
  "overall_risk_level": "LOW|MODERATE|HIGH|CRITICAL",
  "priority_concerns": ["Top concern 1", "Top concern 2", "Top concern 3"],
  "drug_interactions": [
    {{
      "drug_1": "Medication A",
      "drug_2": "Medication B",
      "severity": "mild|moderate|severe",
      "description": "Mechanism of interaction",
      "recommendation": "Clinical action to take"
    }}
  ]
}}

Risk score thresholds: LOW=0-0.3, MODERATE=0.3-0.6, HIGH=0.6-0.8, CRITICAL=0.8-1.0
Only include conditions relevant to this patient. Omit categories with score=0."""

_HUMAN = """Patient Risk Assessment:

Age: {age}
Gender: {gender}
Chief Complaint: {chief_complaint}
Conditions: {conditions}
Current Medications: {medications}
Lab Values: {lab_values}
Vital Signs: {vitals}

Clinical Context (Guidelines Summary):
{guidelines}

Rule-Based Alerts (apply these in your scoring):
{rule_alerts}

Calculate evidence-based risk scores and identify all drug interactions."""

# ── Rule-based engine ─────────────────────────────────────────────────────────

class _RuleEngine:
    """
    Fast, deterministic rule-based risk checks.
    Runs before LLM to provide ground-truth escalation signals.
    """

    ANTICOAGULANTS = {"warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban", "dabigatran", "fondaparinux"}
    NSAIDS = {"ibuprofen", "naproxen", "diclofenac", "celecoxib", "ketorolac", "indomethacin", "meloxicam"}
    CRITICAL_CONDITIONS = {"sepsis", "septic shock", "myocardial infarction", "stroke", "pulmonary embolism", "respiratory failure"}
    NEPHROTOXINS = {"nsaid", "ibuprofen", "naproxen", "gentamicin", "vancomycin", "contrast"}
    DUAL_RAAS = {"ace inhibitor", "acei", "arb", "lisinopril", "ramipril", "losartan", "valsartan", "enalapril"}

    def check(self, conditions: List[str], medications: List[str], lab_values: Dict) -> List[str]:
        alerts: List[str] = []
        conds_lower = {c.lower() for c in conditions}
        meds_lower = {m.lower() for m in medications}

        # Critical condition override
        for cc in self.CRITICAL_CONDITIONS:
            if any(cc in c for c in conds_lower):
                alerts.append(f"CRITICAL CONDITION DETECTED: {cc.title()} — escalate overall risk to CRITICAL")
                break

        # Anticoagulant + NSAID bleeding risk
        has_ac = any(any(ac in m for ac in self.ANTICOAGULANTS) for m in meds_lower)
        has_ns = any(any(ns in m for ns in self.NSAIDS) for m in meds_lower)
        if has_ac and has_ns:
            alerts.append("HIGH BLEEDING RISK: Anticoagulant + NSAID combination detected — 3-4x increased bleeding risk")

        # Dual RAAS blockade
        raas_count = sum(1 for ra in self.DUAL_RAAS if any(ra in m for m in meds_lower))
        if raas_count >= 2:
            alerts.append("DUAL RAAS BLOCKADE: ACE inhibitor + ARB combination — increased hyperkalemia and AKI risk")

        # Metformin + renal impairment
        has_metformin = any("metformin" in m for m in meds_lower)
        egfr = lab_values.get("eGFR", lab_values.get("egfr", ""))
        if has_metformin and egfr:
            try:
                egfr_val = float(str(egfr).split()[0])
                if egfr_val < 30:
                    alerts.append(f"CONTRAINDICATION: Metformin with eGFR={egfr_val} (<30) — lactic acidosis risk, hold medication")
            except ValueError:
                pass

        # Critical lab values
        for lab, val in lab_values.items():
            try:
                num = float(str(val).split()[0].replace(",", ""))
                lab_l = lab.lower()
                if "potassium" in lab_l or lab_l == "k":
                    if num > 5.5:
                        alerts.append(f"CRITICAL LAB: Hyperkalemia K={num} mEq/L — arrhythmia risk, cardiac monitoring required")
                    elif num < 3.0:
                        alerts.append(f"CRITICAL LAB: Hypokalemia K={num} mEq/L — arrhythmia risk, urgent replacement needed")
                if "sodium" in lab_l or lab_l == "na":
                    if num < 125:
                        alerts.append(f"CRITICAL LAB: Severe hyponatremia Na={num} — seizure risk")
                if "glucose" in lab_l:
                    if num > 400:
                        alerts.append(f"CRITICAL LAB: Hyperglycaemic emergency glucose={num} mg/dL")
                    elif num < 50:
                        alerts.append(f"CRITICAL LAB: Severe hypoglycaemia glucose={num} mg/dL")
                if "inr" in lab_l:
                    if num > 4.0:
                        alerts.append(f"CRITICAL LAB: Supratherapeutic INR={num} — major bleeding risk")
                if "lactate" in lab_l:
                    if num >= 4.0:
                        alerts.append(f"CRITICAL LAB: Severe lactic acidosis lactate={num} mmol/L — sepsis/shock indicator")
            except (ValueError, IndexError):
                continue

        return alerts


# ── Agent ─────────────────────────────────────────────────────────────────────

class RiskAgent:
    """Risk Agent: Rule-based pre-checks + LLM risk scoring + drug interaction detection."""

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.0)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM), ("human", _HUMAN)]
        )
        self._rules = _RuleEngine()
        logger.info("RiskAgent initialised")

    @staticmethod
    def _clean_json(raw: str) -> str:
        s = raw.strip()
        if s.startswith("```"):
            inner = s.split("```")[1]
            s = inner[4:].strip() if inner.startswith("json") else inner.strip()
        return s

    def run(self, state: ClinicalState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        trace = state.get("trace_id", "?")
        logger.info(f"RiskAgent.run | trace={trace[:8]}")

        try:
            conditions = state.get("conditions", [])
            medications = state.get("medications", [])
            lab_values = state.get("lab_values", {})
            patient_info = state.get("patient_info", {})
            guidelines = state.get("relevant_guidelines", ["Not available"])

            # Layer 1: Rule-based alerts
            rule_alerts = self._rules.check(conditions, medications, lab_values)
            logger.info(f"RiskAgent: {len(rule_alerts)} rule-based alert(s) triggered")

            # Layer 2: LLM risk scoring
            chain = self._prompt | self._llm
            resp = chain.invoke({
                "age": patient_info.get("age", "Unknown"),
                "gender": patient_info.get("gender", "Unknown"),
                "chief_complaint": state.get("chief_complaint", "Not documented"),
                "conditions": ", ".join(conditions) or "None",
                "medications": ", ".join(medications) or "None",
                "lab_values": json.dumps(lab_values, indent=2) if lab_values else "{}",
                "vitals": json.dumps(patient_info.get("vital_signs", {})),
                "guidelines": guidelines[0][:1200] if guidelines else "N/A",
                "rule_alerts": "\n".join(f"• {a}" for a in rule_alerts) if rule_alerts else "None",
            })

            raw = resp.content if hasattr(resp, "content") else str(resp)
            risk_data: Dict = json.loads(self._clean_json(raw))

            # Hard escalation: if rule engine found critical condition → force CRITICAL
            critical_rules = [a for a in rule_alerts if a.startswith("CRITICAL CONDITION")]
            if critical_rules:
                risk_data["overall_risk_level"] = "CRITICAL"

            risk_scores = risk_data.get("risk_scores", {})
            overall = risk_data.get("overall_risk_level", "MODERATE")
            priority_concerns = risk_data.get("priority_concerns", [])
            drug_interactions = risk_data.get("drug_interactions", [])

            # Merge rule-based alerts into priority concerns
            for alert in rule_alerts:
                priority_concerns.insert(0, alert)

            elapsed = (time.perf_counter() - t0) * 1000
            step = (
                f"✅ Risk Agent ({elapsed:.0f}ms) — "
                f"Overall: {overall} | "
                f"{len(risk_scores)} risk categories | "
                f"{len(drug_interactions)} drug interaction(s) | "
                f"{len(rule_alerts)} rule alert(s)"
            )
            logger.info(step)

            return {
                "risk_scores": risk_scores,
                "risk_level": overall,
                "risk_factors": priority_concerns[:8],
                "drug_interactions": drug_interactions,
                "contraindications": rule_alerts,
                "processing_steps": state.get("processing_steps", []) + [step],
            }

        except Exception as exc:
            logger.exception(f"RiskAgent error: {exc}")
            elapsed = (time.perf_counter() - t0) * 1000
            step = f"⚠️ Risk Agent error ({elapsed:.0f}ms): {exc}"
            return {
                "risk_scores": {},
                "risk_level": "UNKNOWN",
                "risk_factors": [f"Risk assessment error: {exc}"],
                "drug_interactions": [],
                "contraindications": [],
                "processing_steps": state.get("processing_steps", []) + [step],
            }
