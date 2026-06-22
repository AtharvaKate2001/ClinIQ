"""
ClinIQ — Hybrid Retriever
Combines dense semantic search (ChromaDB) + sparse BM25 keyword search,
fused with Reciprocal Rank Fusion (RRF) for best-of-both-worlds retrieval.

This is the key technical differentiator of ClinIQ's RAG system.
Most systems use only semantic search; hybrid retrieval handles both
concept-level questions (semantic) and exact term lookups (BM25).
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from core.config import settings
from core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Two-stage hybrid retriever with Reciprocal Rank Fusion (RRF).

    Stage 1 — Dense retrieval:  ChromaDB cosine similarity
    Stage 2 — Sparse retrieval: BM25 keyword match
    Fusion:   RRF(60) combines both ranked lists without requiring score normalisation

    Reference: Cormack et al. (2009) "Reciprocal rank fusion outperforms condorcet..."
    """

    RRF_K = 60  # standard RRF constant (higher = less sensitivity to top ranks)

    def __init__(self) -> None:
        self._vs = get_vectorstore()
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _build_bm25(self, docs: List[str]) -> None:
        self._bm25_corpus = docs
        self._bm25 = BM25Okapi([self._tokenise(d) for d in docs])

    # ── RRF ───────────────────────────────────────────────────────────────────

    def _rrf(self, *ranked_lists: List[Dict]) -> List[Dict]:
        """
        Merge N ranked result lists via Reciprocal Rank Fusion.
        Each document gets score = Σ 1/(rank + K) across all lists.
        """
        scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc["content"]
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank + self.RRF_K)
                if key not in doc_store:
                    doc_store[key] = doc

        return [
            {**doc_store[k], "rrf_score": round(v, 6)}
            for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        domain_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents using hybrid search.

        Args:
            query:         Natural language query
            k:             Number of results (defaults to settings.RAG_TOP_K)
            domain_filter: Optional metadata filter (e.g. "healthcare")

        Returns:
            List of dicts: {content, metadata, score, rrf_score}
        """
        k = k or settings.RAG_TOP_K
        fetch = k * 3  # over-fetch for RRF re-ranking

        # ── Dense semantic search ─────────────────────────────────────────────
        where = {"domain": domain_filter} if domain_filter else None
        dense_results = self._vs.similarity_search(query, k=fetch, where=where)

        # ── BM25 sparse search ────────────────────────────────────────────────
        sparse_results: List[Dict] = []
        if self._bm25 and self._bm25_corpus:
            q_tokens = self._tokenise(query)
            bm25_scores = self._bm25.get_scores(q_tokens)
            ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            for idx in ranked_indices[:fetch]:
                if bm25_scores[idx] > 0:
                    sparse_results.append(
                        {"content": self._bm25_corpus[idx], "metadata": {}, "score": float(bm25_scores[idx])}
                    )

        # ── Fusion ────────────────────────────────────────────────────────────
        if sparse_results:
            fused = self._rrf(dense_results, sparse_results)
        else:
            fused = dense_results  # only semantic available

        logger.debug(
            f"Hybrid retrieval: {len(fused)} docs fused "
            f"(dense={len(dense_results)}, sparse={len(sparse_results)}) "
            f"for query: '{query[:60]}'"
        )
        return fused[:k]

    def update_bm25_index(self, documents: List[str]) -> None:
        """Rebuild BM25 index from a list of raw document strings."""
        self._build_bm25(documents)
        logger.info(f"BM25 index rebuilt with {len(documents)} documents")


# ── Singleton ─────────────────────────────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
