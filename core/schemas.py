"""
ClinIQ — Pydantic Schemas
Request/response models for the FastAPI REST API.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


# ── Request Models ────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    document: str = Field(..., min_length=10, description="Raw medical document text")
    document_type: str = Field(default="medical_report")
    patient_id: Optional[str] = Field(default=None)
    enable_drug_check: bool = Field(default=True)
    session_id: Optional[str] = Field(default=None)

    model_config = {"json_schema_extra": {
        "example": {
            "document": "Patient: John Smith, 67M. Chief complaint: Dyspnea...",
            "document_type": "discharge_summary",
            "enable_drug_check": True,
        }
    }}


class KnowledgeBaseIngestRequest(BaseModel):
    """Request body for /knowledge-base/ingest endpoint."""
    text: str = Field(..., min_length=20)
    source: str = Field(..., description="e.g. 'AHA Guidelines 2024'")
    category: str = Field(
        ...,
        description="guideline | drug_info | condition | procedure | reference"
    )
    tags: List[str] = Field(default_factory=list)


# ── Nested Response Models ────────────────────────────────────────────────────

class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_type: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    allergies: List[str] = Field(default_factory=list)


class RiskScoreDetail(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: RiskLevel
    factors: List[str] = Field(default_factory=list)
    justification: str = ""


class DrugInteraction(BaseModel):
    drug_1: str
    drug_2: str
    severity: str  # mild | moderate | severe
    description: str
    recommendation: str


class ClinicalFinding(BaseModel):
    category: str
    finding: str
    significance: str  # normal | abnormal | critical
    recommendation: Optional[str] = None


# ── Top-level Response ────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Full response from the /analyze endpoint."""
    trace_id: str
    status: str = "success"
    processing_time_seconds: float

    # Structured patient data (Intake Agent)
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    lab_values: Dict[str, Any] = Field(default_factory=dict)
    chief_complaint: str = ""

    # Risk assessment (Risk Agent)
    risk_scores: Dict[str, Any] = Field(default_factory=dict)
    overall_risk_level: RiskLevel = RiskLevel.UNKNOWN
    drug_interactions: List[Dict[str, Any]] = Field(default_factory=list)

    # RAG output (Knowledge Agent)
    retrieved_knowledge_count: int = 0
    guidelines_summary: str = ""

    # Final report (Report Agent)
    final_report: Dict[str, Any] = Field(default_factory=dict)

    # Pipeline audit
    processing_steps: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeBaseStats(BaseModel):
    total_documents: int
    embedding_model: str
    collection_name: str
    domain: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_provider: str
    domain: str
    knowledge_base_docs: int
