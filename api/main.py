"""
ClinIQ — FastAPI Application
Production REST API with:
  GET  /health                  — Health check + system status
  POST /analyze                 — Core multi-agent analysis pipeline
  POST /upload-document         — Parse PDF/DOCX to text
  POST /knowledge-base/ingest   — Add document to knowledge base
  GET  /knowledge-base/stats    — Knowledge base metrics

Run locally:
  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.observability import get_observability
from core.schemas import (
    AnalysisRequest,
    KnowledgeBaseIngestRequest,
)
from core.vectorstore import get_vectorstore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all singletons on startup; flush observability on shutdown."""
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"   LLM Provider : {settings.LLM_PROVIDER}")
    logger.info(f"   Domain       : {settings.DOMAIN}")

    # Vector store — triggers embedding model download on first run
    vs = get_vectorstore()
    doc_count = vs.count()
    logger.info(f"   Knowledge base: {doc_count} documents")

    # Auto-seed if empty
    if doc_count == 0:
        logger.info("   Knowledge base empty — seeding medical knowledge...")
        loop = asyncio.get_event_loop()
        try:
            from tools.knowledge_seeder import seed_medical_knowledge
            await loop.run_in_executor(None, seed_medical_knowledge)
        except Exception as exc:
            logger.warning(f"   Seed warning: {exc}")

    # Pre-warm the LangGraph workflow (compiles nodes on first call)
    try:
        from agents.graph import get_workflow
        _ = get_workflow()
        logger.info("   LangGraph workflow: compiled ✅")
    except Exception as exc:
        logger.error(f"   Workflow compile error: {exc}")

    # Observability
    obs = get_observability()
    logger.info(f"   Observability: {'Langfuse' if obs._langfuse else 'structured logs'} ✅")

    logger.info("✅ ClinIQ API ready\n")
    yield

    # Shutdown
    get_observability().flush()
    logger.info("👋 ClinIQ API stopped")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Autonomous Multi-Agent Clinical Intelligence Platform. "
        "Drop in a medical document — get structured clinical insights, "
        "risk scores, drug interactions, and a full AI-generated report."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """System health and configuration overview."""
    vs = get_vectorstore()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "domain": settings.DOMAIN,
        "knowledge_base_docs": vs.count(),
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_document(request: AnalysisRequest):
    """
    Core endpoint — runs the full 4-agent LangGraph pipeline.

    Pipeline:
    1. Intake Agent      — Extract structured patient data (LLM)
    2. Knowledge Agent   — Hybrid RAG over medical knowledge base
    3. Risk Agent        — Rule-based + LLM risk scoring + drug interactions
    4. Report Agent      — Structured clinical intelligence report

    Returns a complete clinical analysis including risk scores, drug interactions,
    guideline summaries, and a narrative report — all in a single JSON response.
    """
    start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    obs = get_observability()

    obs.create_trace(
        name="clinical_analysis",
        metadata={
            "trace_id": trace_id,
            "document_length": len(request.document),
            "patient_id": request.patient_id,
            "domain": settings.DOMAIN,
        },
    )

    logger.info(
        f"Analysis request | trace={trace_id[:8]} | "
        f"doc_len={len(request.document)} chars | type={request.document_type}"
    )

    try:
        from agents.graph import get_workflow

        workflow = get_workflow()

        initial_state = {
            "raw_document": request.document,
            "document_type": request.document_type,
            "trace_id": trace_id,
            "messages": [],
            "processing_steps": [f"🔵 Pipeline started | trace: {trace_id}"],
            # Pre-initialise all state keys to avoid TypedDict KeyErrors
            "patient_info": {},
            "conditions": [],
            "medications": [],
            "lab_values": {},
            "chief_complaint": "",
            "retrieved_knowledge": [],
            "relevant_guidelines": [],
            "risk_scores": {},
            "risk_factors": [],
            "risk_level": "UNKNOWN",
            "drug_interactions": [],
            "contraindications": [],
            "final_report": {},
            "error": None,
        }

        config = {
            "configurable": {"thread_id": trace_id},
            "recursion_limit": 12,
        }

        # Run workflow in executor (blocking LLM calls → async-safe)
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: workflow.invoke(initial_state, config=config),
        )

        elapsed = time.perf_counter() - start

        response = {
            "trace_id": trace_id,
            "status": "success",
            "processing_time_seconds": round(elapsed, 2),
            # Intake
            "patient_info": final_state.get("patient_info", {}),
            "conditions": final_state.get("conditions", []),
            "medications": final_state.get("medications", []),
            "lab_values": final_state.get("lab_values", {}),
            "chief_complaint": final_state.get("chief_complaint", ""),
            # Risk
            "risk_scores": final_state.get("risk_scores", {}),
            "overall_risk_level": final_state.get("risk_level", "UNKNOWN"),
            "drug_interactions": final_state.get("drug_interactions", []),
            "risk_factors": final_state.get("risk_factors", []),
            # Knowledge
            "retrieved_knowledge_count": len(final_state.get("retrieved_knowledge", [])),
            "guidelines_summary": (
                final_state["relevant_guidelines"][0][:1200]
                if final_state.get("relevant_guidelines")
                else ""
            ),
            # Report
            "final_report": final_state.get("final_report", {}),
            # Audit
            "processing_steps": final_state.get("processing_steps", []),
        }

        obs.log_span(
            trace_id=trace_id,
            name="pipeline_complete",
            output_data={"risk_level": response["overall_risk_level"]},
            duration_ms=elapsed * 1000,
        )

        logger.info(
            f"Analysis complete | trace={trace_id[:8]} | "
            f"risk={response['overall_risk_level']} | "
            f"time={elapsed:.2f}s"
        )
        return response

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception(f"Analysis failed | trace={trace_id[:8]} | error={exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(exc)}",
        ) from exc


@app.post("/upload-document", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and parse a clinical document (PDF, DOCX, TXT).

    Returns extracted plain text ready to pass to /analyze.
    Max recommended size: 10MB.
    """
    ALLOWED = {
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: PDF, TXT, DOCX",
        )

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:  # 15 MB hard cap
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    from tools.document_parser import DocumentParser

    try:
        text = DocumentParser().parse_bytes(content, file.content_type, filename=file.filename or "")
        return {
            "status": "success",
            "filename": file.filename,
            "content_type": file.content_type,
            "extracted_text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Document parsing failed: {str(exc)}") from exc


@app.post("/knowledge-base/ingest", tags=["Knowledge Base"])
async def ingest_knowledge(request: KnowledgeBaseIngestRequest):
    """Add a new document to the medical knowledge base (vector store)."""
    vs = get_vectorstore()
    doc_id = f"user_{str(uuid.uuid4())[:8]}"

    vs.add_documents(
        documents=[request.text],
        metadatas=[
            {
                "source": request.source,
                "category": request.category,
                "tags": ", ".join(request.tags),
                "domain": settings.DOMAIN,
            }
        ],
        ids=[doc_id],
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "total_documents": vs.count(),
    }


@app.get("/knowledge-base/stats", tags=["Knowledge Base"])
async def knowledge_base_stats():
    """Knowledge base metrics."""
    vs = get_vectorstore()
    return {
        "total_documents": vs.count(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "collection_name": settings.CHROMA_COLLECTION_NAME,
        "domain": settings.DOMAIN,
        "persist_dir": settings.CHROMA_PERSIST_DIR,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
