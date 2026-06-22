"""
ClinIQ — Document Parser
Handles PDF, DOCX, and plain text extraction.
Uses PyMuPDF (fitz) as primary PDF engine with graceful fallback.
"""
from __future__ import annotations
import io
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Multi-format document parser supporting:
      PDF  → PyMuPDF (fitz) → pdfplumber fallback
      DOCX → python-docx
      TXT  → UTF-8 decode
    """

    SUPPORTED_TYPES = {
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_bytes(
        self,
        content: bytes,
        content_type: str,
        filename: str = "",
    ) -> str:
        """Parse document bytes into plain text."""
        fname = filename.lower()

        if content_type == "application/pdf" or fname.endswith(".pdf"):
            return self._parse_pdf(content)

        if content_type == "text/plain" or fname.endswith(".txt") or fname.endswith(".md"):
            return content.decode("utf-8", errors="replace")

        if "word" in content_type or fname.endswith(".docx") or fname.endswith(".doc"):
            return self._parse_docx(content)

        # Last resort: try UTF-8 decode
        logger.warning(f"Unknown content type '{content_type}' — attempting UTF-8 decode")
        return content.decode("utf-8", errors="replace")

    def parse_file(self, filepath: str) -> str:
        """Parse a file from disk into plain text."""
        path = Path(filepath)
        content = path.read_bytes()
        content_type = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/plain",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }.get(path.suffix.lower(), "text/plain")
        return self.parse_bytes(content, content_type, filename=path.name)

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _parse_pdf(self, content: bytes) -> str:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=content, filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n".join(pages)
            logger.info(f"PDF parsed via PyMuPDF: {len(text)} chars")
            return self.clean(text)

        except ImportError:
            logger.warning("PyMuPDF not installed — trying pdfplumber")
            return self._parse_pdf_pdfplumber(content)

        except Exception as exc:
            logger.error(f"PyMuPDF error: {exc}")
            raise RuntimeError(f"PDF parsing failed: {exc}") from exc

    def _parse_pdf_pdfplumber(self, content: bytes) -> str:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages)
            logger.info(f"PDF parsed via pdfplumber: {len(text)} chars")
            return self.clean(text)

        except ImportError:
            raise RuntimeError(
                "No PDF library available. "
                "Install: pip install pymupdf  (or pip install pdfplumber)"
            )

    # ── DOCX ──────────────────────────────────────────────────────────────────

    def _parse_docx(self, content: bytes) -> str:
        try:
            from docx import Document

            doc = Document(io.BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(paragraphs)
            logger.info(f"DOCX parsed: {len(text)} chars")
            return self.clean(text)

        except ImportError:
            raise RuntimeError(
                "python-docx not installed. Install: pip install python-docx"
            )

    # ── Cleaning ──────────────────────────────────────────────────────────────

    @staticmethod
    def clean(text: str) -> str:
        """Normalise whitespace in extracted text."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\x00", "", text)  # null bytes from some PDFs
        return text.strip()
