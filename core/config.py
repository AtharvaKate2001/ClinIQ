"""
ClinIQ — Configuration
Pydantic Settings-based configuration with environment variable support.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
import os


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "ClinIQ"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Priority: groq (free, fast) → ollama (local, free) → google → openai
    LLM_PROVIDER: Literal["groq", "ollama", "openai", "google"] = "groq"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama3-8b-8192"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GOOGLE_API_KEY: str = ""
    GOOGLE_MODEL: str = "gemini-1.5-flash"

    # ── Embeddings (local, free) ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "cliniq_knowledge"

    # ── Observability ─────────────────────────────────────────────────────────
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ── API Server ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Domain ────────────────────────────────────────────────────────────────
    # Switch to "finance" to adapt for financial document analysis
    DOMAIN: Literal["healthcare", "finance"] = "healthcare"

    # ── RAG ───────────────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.3

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


settings = Settings()
