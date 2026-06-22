"""
ClinIQ — LLM Factory
Provides a unified interface to swap between free LLM providers.
Priority: Groq (free) → Ollama (local/free) → Google (free tier) → OpenAI
"""
from __future__ import annotations
import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel
from core.config import settings

logger = logging.getLogger(__name__)


def get_llm(
    temperature: float = 0.1,
    use_fast_model: bool = False,
    model_override: Optional[str] = None,
) -> BaseChatModel:
    """
    Return the configured LLM with automatic provider fallback.

    Args:
        temperature:    Sampling temperature (0 = deterministic, 1 = creative)
        use_fast_model: Use smaller/faster model for quick extraction tasks
        model_override: Force a specific model name

    Returns:
        Configured LangChain chat model instance

    Provider config (set LLM_PROVIDER in .env):
      groq   → Groq cloud API (FREE, 14k TPM on free tier)
      ollama → Local Ollama (FREE, requires ollama running)
      google → Google Gemini (FREE tier available)
      openai → OpenAI (paid)
    """
    provider = settings.LLM_PROVIDER

    # ── Groq (recommended free provider) ──────────────────────────────────────
    if provider == "groq" and settings.GROQ_API_KEY:
        from langchain_groq import ChatGroq

        model = model_override or (
            settings.GROQ_FAST_MODEL if use_fast_model else settings.GROQ_MODEL
        )
        logger.info(f"LLM: Groq / {model}")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=4096,
        )

    # ── Ollama (local, 100% free) ─────────────────────────────────────────────
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            model = model_override or settings.OLLAMA_MODEL
            logger.info(f"LLM: Ollama / {model} @ {settings.OLLAMA_BASE_URL}")
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=model,
                temperature=temperature,
            )
        except ImportError:
            logger.warning("langchain-ollama not installed; trying next provider")

    # ── Google Gemini (free tier) ─────────────────────────────────────────────
    if provider == "google" and settings.GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = model_override or settings.GOOGLE_MODEL
            logger.info(f"LLM: Google / {model}")
            return ChatGoogleGenerativeAI(
                google_api_key=settings.GOOGLE_API_KEY,
                model=model,
                temperature=temperature,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed")

    # ── OpenAI (paid) ─────────────────────────────────────────────────────────
    if provider == "openai" and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI

            model = model_override or settings.OPENAI_MODEL
            logger.info(f"LLM: OpenAI / {model}")
            return ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=model,
                temperature=temperature,
            )
        except ImportError:
            logger.warning("langchain-openai not installed")

    # ── Hard fallback: Ollama (always free) ───────────────────────────────────
    logger.warning("No matching provider found — falling back to local Ollama. "
                   "Run: ollama pull llama3.2")
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model="llama3.2",
            temperature=temperature,
        )
    except ImportError:
        raise RuntimeError(
            "No LLM provider available! Set GROQ_API_KEY in .env "
            "or install langchain-ollama and run Ollama locally."
        )


def get_fast_llm(temperature: float = 0.0) -> BaseChatModel:
    """Convenience wrapper for quick extraction tasks (smaller, faster model)."""
    return get_llm(temperature=temperature, use_fast_model=True)
