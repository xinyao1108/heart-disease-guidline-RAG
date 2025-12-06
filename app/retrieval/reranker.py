"""Cross-encoder reranker backed by FlagEmbedding."""

from __future__ import annotations

from typing import List

from FlagEmbedding import FlagReranker

from app.models.retrieval import RetrievedChunk

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Applies cross-encoder reranking to retrieved candidates."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = FlagReranker(model_name, use_fp16=False, devices="cpu")

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 10) -> List[RetrievedChunk]:
        if not candidates:
            return []
        sentence_pairs = [(query, chunk.text) for chunk in candidates]
        scores = self.model.compute_score(sentence_pairs)
        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
        reranked = sorted(candidates, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]
