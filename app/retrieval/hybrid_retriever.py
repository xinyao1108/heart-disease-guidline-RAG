"""Hybrid retriever that combines BM25 and dense search with RRF."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

from app.models.retrieval import RetrievedChunk
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import embed_queries
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 50


class HybridRetriever:
    """Combines sparse, dense, and reranking stages."""

    def __init__(
        self,
        bm25_store: BM25Store | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.bm25_store = bm25_store or BM25Store()
        self.vector_store = vector_store or VectorStore()

    def _rrf_merge(
        self,
        sparse_results: Iterable[RetrievedChunk],
        dense_results: Iterable[RetrievedChunk],
    ) -> Dict[str, RetrievedChunk]:
        fused: Dict[str, RetrievedChunk] = {}

        def apply_rrf(candidates: Iterable[RetrievedChunk], attr: str) -> None:
            for rank, chunk in enumerate(candidates, start=1):
                existing = fused.get(chunk.chunk_id)
                if not existing:
                    existing = chunk.model_copy(deep=True)
                    fused[chunk.chunk_id] = existing
                setattr(existing, attr, getattr(chunk, attr))
                existing.fused_score = (existing.fused_score or 0.0) + 1.0 / (RRF_K + rank)

        apply_rrf(sparse_results, "sparse_score")
        apply_rrf(dense_results, "dense_score")
        return fused

    def retrieve(
        self,
        question: str,
        top_k_sparse: int = 32,
        top_k_dense: int = 32,
        top_k_final: int = 20,
    ) -> List[RetrievedChunk]:
        sparse_hits: List[RetrievedChunk] = []
        dense_hits: List[RetrievedChunk] = []

        if self.bm25_store:
            sparse_hits = self.bm25_store.search(question, top_k=top_k_sparse)
        else:
            logger.warning("BM25 store unavailable; skipping sparse retrieval.")

        try:
            query_vector = embed_queries([question])[0]
            dense_hits = self.vector_store.search(query_vector, top_k=top_k_dense)
        except Exception as exc:  # pragma: no cover - safety net
            logger.error("Dense retrieval failed: %s", exc)

        fused = self._rrf_merge(sparse_hits, dense_hits)
        ranked = sorted(
            fused.values(),
            key=lambda chunk: chunk.fused_score or 0.0,
            reverse=True,
        )
        return ranked[:top_k_final]
