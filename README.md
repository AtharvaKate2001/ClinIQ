![ClinIQ ](assets/cliniq.png)

# 🏥 ClinIQ — Autonomous Multi-Agent Clinical Intelligence Platform

> Drop in any clinical document. Get AI-powered risk analysis, drug interaction checks, evidence-based guidelines, and a fully structured clinical report — powered by a LangGraph multi-agent pipeline.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.114+-green)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-red)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What Makes ClinIQ Unique

Most RAG/agent projects use a single LLM call with document retrieval. ClinIQ is architecturally different:

| Feature | Typical RAG Project | ClinIQ |
|---------|---------------------|--------|
| Architecture | Single chain | LangGraph stateful multi-agent |
| Retrieval | Semantic only | Hybrid BM25 + Semantic + RRF fusion |
| Risk Scoring | LLM only | Deterministic rules + LLM (two-layer) |
| Observability | None | Langfuse LLM tracing |
| State Management | None | LangGraph with checkpointing |
| Domain Adaptation | Hard-coded | Config-driven (`DOMAIN=finance`) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ClinIQ LangGraph Pipeline                   │
│                                                                  │
│  User/API ──► [FastAPI] ──► [LangGraph StateGraph]             │
│                                      │                           │
│              ┌───────────────────────▼────────────────────────┐ │
│              │         ClinicalState (TypedDict)               │ │
│              └──────────────────────────────────────────────── │ │
│                                      │                           │
│                    ┌─────────────────▼─────────────────────┐   │
│                    │      [1] Intake Agent                  │   │
│                    │  LLM → Structured JSON extraction       │   │
│                    │  Extracts: patient_info, conditions,    │   │
│                    │  medications, lab_values, vitals        │   │
│                    └─────────────────┬─────────────────────┘   │
│                                      │ (error → skip to report)  │
│                    ┌─────────────────▼─────────────────────┐   │
│                    │      [2] Knowledge Agent               │   │
│                    │  Multi-query Hybrid RAG                 │   │
│                    │  BM25 + Semantic + RRF Fusion           │   │
│                    │  → Guideline synthesis (LLM)            │   │
│                    └─────────────────┬─────────────────────┘   │
│                                      │                           │
│                    ┌─────────────────▼─────────────────────┐   │
│                    │      [3] Risk Agent                    │   │
│                    │  Layer 1: Rule Engine (deterministic)   │   │
│                    │    - Anticoagulant + NSAID combos       │   │
│                    │    - Dual RAAS blockade                  │   │
│                    │    - Critical lab thresholds             │   │
│                    │    - Metformin + CKD contraindication    │   │
│                    │  Layer 2: LLM Risk Scoring              │   │
│                    │    - 30-day readmission, CV, renal,     │   │
│                    │      medication adverse event           │   │
│                    │  + Drug interaction detection (LLM)     │   │
│                    └─────────────────┬─────────────────────┘   │
│                                      │                           │
│                    ┌─────────────────▼─────────────────────┐   │
│                    │      [4] Report Agent                  │   │
│                    │  Synthesis → Structured JSON Report     │   │
│                    │  executive_summary + clinical_findings  │   │
│                    │  + recommendations + follow_up          │   │
│                    │  + clinical_narrative (record-quality)  │   │
│                    └─────────────────┬─────────────────────┘   │
│                                      │                           │
│              JSON Response ◄─────────┘                           │
│              (+ Langfuse trace)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack (All Free)

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent Orchestration | **LangGraph** | Stateful agents, conditional edges, memory |
| LLM | **Groq** (Llama 3.3 70B) | FREE, fast (600 tokens/sec) |
| Fallback LLM | **Ollama** (Llama 3.2) | 100% local, completely free |
| Vector Store | **ChromaDB** | Local persistent, no API key needed |
| Embeddings | **sentence-transformers** | Local inference, free |
| Hybrid Retrieval | **BM25 + Cosine + RRF** | Best-of-both retrieval |
| Observability | **Langfuse** | Free cloud tier — LLM tracing |
| API | **FastAPI** | Async, auto-docs, production-grade |
| Frontend | **Streamlit + Plotly** | Interactive UI with charts |
| Containers | **Docker + Compose** | One-command deployment |

---

## 📁 Project Structure

```
cliniq/
├── README.md
├── requirements.txt
├── .env.example                ← Copy to .env, add GROQ_API_KEY
├── docker-compose.yml
├── Dockerfile                  ← FastAPI backend
├── Dockerfile.frontend         ← Streamlit frontend
├── Makefile                    ← Dev shortcuts
│
├── core/
│   ├── config.py               ← Pydantic Settings (all env vars)
│   ├── state.py                ← LangGraph TypedDict ClinicalState
│   ├── llm_factory.py          ← LLM factory: Groq → Ollama → Google → OpenAI
│   ├── vectorstore.py          ← ChromaDB + sentence-transformers
│   ├── retriever.py            ← Hybrid BM25 + Semantic + RRF
│   ├── schemas.py              ← Pydantic API request/response models
│   └── observability.py        ← Langfuse + structured logging
│
├── agents/
│   ├── graph.py                ← LangGraph StateGraph workflow
│   ├── intake_agent.py         ← Document parsing & structuring
│   ├── knowledge_agent.py      ← RAG retrieval & synthesis
│   ├── risk_agent.py           ← Risk scoring (rules + LLM) + drug checks
│   └── report_agent.py         ← Final report generation
│
├── tools/
│   ├── document_parser.py      ← PDF/DOCX/TXT parser (PyMuPDF)
│   └── knowledge_seeder.py     ← Seeds 8 medical knowledge chunks
│
├── api/
│   └── main.py                 ← FastAPI app (4 endpoints)
│
├── frontend/
│   └── app.py                  ← Streamlit UI (gauge, charts, tabs)
│
└── data/
    ├── chroma_db/              ← Auto-created (ChromaDB persistent storage)
    ├── checkpoints/            ← LangGraph state checkpoints
    └── samples/
        ├── sample_patient_report.txt
        └── sample_finance_adaptation.txt
```

---

## 🚀 Quick Start

### Option A: Run Locally (Recommended for Development)

```bash
# 1. Clone the repository
git clone https://github.com/AtharvaKate2001/cliniq
cd cliniq

# 2. Set up environment
make setup
# OR manually:
pip install -r requirements.txt
cp .env.example .env

# 3. Get your FREE Groq API key
# → https://console.groq.com (free signup, no credit card)
# Edit .env and add: GROQ_API_KEY=your_key_here

# 4. Start the FastAPI backend
make run-api
# → http://localhost:8000/docs

# 5. Start the Streamlit frontend (new terminal)
make run-frontend
# → http://localhost:8501
```

### Option B: Docker (One-command deployment)

```bash
cp .env.example .env
# Edit .env: add GROQ_API_KEY=...
docker compose up --build
# → API: http://localhost:8000
# → UI:  http://localhost:8501
```

### Option C: 100% Free (No API key — use Ollama)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2

# In .env:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2

make run-api
make run-frontend
```

---

## 🔑 API Reference

### `POST /analyze`
Run the full 4-agent analysis pipeline.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Patient: John Smith, 67M. Chief complaint: Dyspnea...",
    "document_type": "discharge_summary",
    "enable_drug_check": true
  }'
```

**Response:**
```json
{
  "trace_id": "uuid",
  "status": "success",
  "processing_time_seconds": 8.4,
  "overall_risk_level": "HIGH",
  "conditions": ["Heart Failure (HFrEF)", "Type 2 Diabetes"],
  "medications": ["Warfarin 5mg", "Ibuprofen 400mg PRN"],
  "risk_scores": {
    "30_day_readmission": {"score": 0.78, "level": "HIGH", "factors": [...], "justification": "..."},
    "cardiovascular": {"score": 0.82, "level": "CRITICAL", ...}
  },
  "drug_interactions": [
    {"drug_1": "Warfarin", "drug_2": "Ibuprofen", "severity": "severe", ...}
  ],
  "final_report": {
    "executive_summary": "...",
    "recommendations": [...],
    "clinical_narrative": "..."
  },
  "processing_steps": ["✅ Intake Agent (1843ms)...", "✅ Knowledge Agent..."]
}
```

### `POST /upload-document`
Upload PDF/DOCX and get extracted text.

```bash
curl -X POST http://localhost:8000/upload-document \
  -F "file=@patient_report.pdf"
```

### `GET /health`
```bash
curl http://localhost:8000/health
```

### `POST /knowledge-base/ingest`
Add to the medical knowledge base.

```bash
curl -X POST http://localhost:8000/knowledge-base/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "source": "AHA 2024", "category": "guideline", "tags": ["cardiology"]}'
```

---

## 🔄 Finance Domain Adaptation

ClinIQ is domain-agnostic. To adapt for financial document analysis:

**1. Change configuration (`.env`):**
```bash
DOMAIN=finance
```

**2. Replace knowledge base** in `tools/knowledge_seeder.py` with:
- Basel III/IV capital adequacy requirements
- IFRS 9 expected credit loss (ECL) model
- Altman Z-Score and financial distress prediction
- Sector-specific leverage and coverage benchmarks
- Regulatory capital frameworks (BIS, FCA, RBI)

**3. The agents automatically adapt:**
| Agent | Healthcare | Finance |
|-------|-----------|---------|
| Intake | Patient info, labs, medications | Company financials, debt structure, covenants |
| Knowledge | Clinical guidelines | Credit guidelines, ratio benchmarks |
| Risk | Clinical risk scores | Credit risk, refinancing risk, liquidity risk |
| Report | Clinical narrative | Credit committee memo |

See `data/samples/sample_finance_adaptation.txt` for a complete example.

---

## 📊 Observability (Optional — Free)

ClinIQ integrates with **Langfuse** for LLM observability:

```bash
# Sign up free: https://cloud.langfuse.com
# Add to .env:
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
```

What you get:
- Full trace for every analysis run
- Token usage per agent
- Latency breakdown by LLM call
- Cost tracking
- Prompt versioning

If keys are not set, ClinIQ falls back to structured logging (still fully functional).

---

## 🧪 Testing

```bash
# Test API health
curl http://localhost:8000/health

# Test with sample file
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @- < <(jq -n --rawfile doc data/samples/sample_patient_report.txt \
    '{"document": $doc, "document_type": "discharge_summary"}')
```

---

## 🏆 Key Technical Concepts (for interviews)

**Why LangGraph over plain LangChain?**
LangGraph provides a proper state machine with typed state, conditional edges, checkpointing, and the ability to resume interrupted workflows. This is how production agentic systems are built, not simple `chain.invoke()`.

**Why Hybrid Retrieval (BM25 + Semantic + RRF)?**
- Semantic search excels at concept-level queries ("medications for heart failure")
- BM25 excels at exact term lookups ("lisinopril", "LVEF 28%", "INR 4.1")
- RRF combines both without requiring score normalisation, giving best-of-both results
- Real clinical documents need both — drug names are exact terms, conditions are conceptual

**Why two-layer risk scoring?**
- Deterministic rules catch known-dangerous combinations with 100% reliability (e.g. anticoagulant + NSAID)
- LLM handles nuanced, multi-factorial clinical reasoning that rules can't capture
- Combining both prevents the LLM from "forgetting" obvious high-severity interactions

**Why Langfuse observability?**
Production LLM applications need tracing, not just logging. Langfuse gives you token usage per agent, latency breakdown, and the ability to replay conversations — critical for debugging and cost optimization.

---

## 📝 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | — | Groq API key (free at console.groq.com) |
| `LLM_PROVIDER` | No | `groq` | `groq \| ollama \| openai \| google` |
| `DOMAIN` | No | `healthcare` | `healthcare \| finance` |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Local sentence-transformer model |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse observability (optional) |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse observability (optional) |

*Not required if using `LLM_PROVIDER=ollama`

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

```bash
git checkout -b feature/your-feature
# make changes
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👨‍💻 Author

**Atharva Kate** — AI Engineer  

---

*Built with LangGraph · Groq · ChromaDB · FastAPI · Streamlit*
