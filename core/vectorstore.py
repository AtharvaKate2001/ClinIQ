"""
ClinIQ — Vector Store Manager
ChromaDB with sentence-transformers embeddings (100% local, free).
Handles document storage, retrieval, and cosine similarity search.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB persistent vector store with sentence-transformers embeddings.

    Design:
    - Embeddings generated locally (sentence-transformers) — no API cost
    - ChromaDB persists to disk (./data/chroma_db)
    - Cosine similarity search with configurable score threshold
    - Thread-safe singleton via module-level get_vectorstore()
    """

    def __init__(self) -> None:
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self._embed_model = SentenceTransformer(
            settings.EMBEDDING_MODEL, trust_remote_code=False
        )

        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"VectorStore ready — collection: '{settings.CHROMA_COLLECTION_NAME}', "
            f"documents: {self._collection.count()}"
        )

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate L2-normalised embeddings (required for cosine similarity)."""
        vecs = self._embed_model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """Upsert documents into the collection (no duplicates on re-run)."""
        if not documents:
            return
        embeddings = self.embed(documents)
        self._collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Upserted {len(documents)} documents into vector store")

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k documents by cosine similarity.

        Returns:
            List of dicts: {content, metadata, score}  (score: 0-1, higher = better)
        """
        total = self._collection.count()
        if total == 0:
            return []

        n = min(k, total)
        q_vec = self.embed([query])[0]

        results = self._collection.query(
            query_embeddings=[q_vec],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1.0 - dist  # ChromaDB returns distance; convert to similarity
                if score >= settings.RAG_SCORE_THRESHOLD:
                    hits.append({"content": doc, "metadata": meta, "score": round(score, 4)})

        return hits

    def count(self) -> int:
        return self._collection.count()


# ── Singleton ─────────────────────────────────────────────────────────────────
_vs: Optional[VectorStoreManager] = None


def get_vectorstore() -> VectorStoreManager:
    """Return the module-level singleton VectorStoreManager."""
    global _vs
    if _vs is None:
        _vs = VectorStoreManager()
    return _vs
