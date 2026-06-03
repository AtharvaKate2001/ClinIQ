"""
ClinIQ — Observability
Unified observability layer:
  - Langfuse (if keys configured): full LLM trace/span tracking
  - Structured logging (always): JSON-friendly, searchable logs

Langfuse is free at cloud.langfuse.com — sign up for the dashboard.
"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Dict, Optional

from core.config import settings

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """
    Single observability facade.
    Write your code against this class; backend (Langfuse / logs) is swappable.
    """

    def __init__(self) -> None:
        self._langfuse: Any = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse(
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                logger.info("✅ Langfuse observability initialised")
            except Exception as exc:
                logger.warning(f"Langfuse init failed ({exc}); using structured logs only")
        else:
            logger.info("📝 Langfuse keys not set — structured logging only. "
                        "Sign up free at cloud.langfuse.com to enable full LLM tracing.")

    # ── Traces ────────────────────────────────────────────────────────────────

    def create_trace(self, name: str, metadata: Optional[Dict] = None) -> str:
        """Create a new observability trace and return its ID."""
        trace_id = str(uuid.uuid4())
        meta = metadata or {}

        if self._langfuse:
            try:
                self._langfuse.trace(id=trace_id, name=name, metadata=meta)
            except Exception as exc:
                logger.warning(f"Langfuse trace creation failed: {exc}")

        logger.info(f"TRACE_START | id={trace_id} | name={name} | meta={meta}")
        return trace_id

    # ── Spans ─────────────────────────────────────────────────────────────────

    def log_span(
        self,
        trace_id: str,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[Dict] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        if self._langfuse:
            try:
                self._langfuse.span(
                    trace_id=trace_id,
                    name=name,
                    input=str(input_data)[:500] if input_data else None,
                    output=str(output_data)[:500] if output_data else None,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning(f"Langfuse span failed: {exc}")

        dur = f" | duration={duration_ms:.1f}ms" if duration_ms is not None else ""
        logger.info(f"SPAN | trace={trace_id[:8]} | name={name}{dur}")

    # ── LLM Calls ─────────────────────────────────────────────────────────────

    def log_llm_call(
        self,
        trace_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        total_tokens = prompt_tokens + completion_tokens
        logger.info(
            f"LLM_CALL | trace={trace_id[:8]} | model={model} "
            f"| prompt_tok={prompt_tokens} | compl_tok={completion_tokens} "
            f"| total_tok={total_tokens} | latency={latency_ms:.0f}ms"
        )

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception:
                pass


# ── Singleton ─────────────────────────────────────────────────────────────────
_obs: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    global _obs
    if _obs is None:
        _obs = ObservabilityManager()
    return _obs
