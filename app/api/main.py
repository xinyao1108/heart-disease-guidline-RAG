"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.llm.answer_generator import AnswerGenerator
from app.models.qa import QARequest, QAResponse
from app.retrieval.evidence import build_evidence_blocks
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedAgenticSystem",
    description="Heart disease guideline RAG API",
    version="0.1.0",
)

retriever = HybridRetriever()
reranker = Reranker()
answer_generator = AnswerGenerator()


@app.get("/health")
def health() -> dict[str, str]:
    """Simple readiness probe."""
    return {"status": "ok"}


@app.post("/ask", response_model=QAResponse)
async def ask(payload: QARequest) -> QAResponse:
    """Answer a clinician question using guideline evidence."""
    candidates = retriever.retrieve(payload.question)
    reranked = reranker.rerank(payload.question, candidates, top_k=10)
    if not reranked:
        raise HTTPException(status_code=404, detail="No relevant guideline evidence found.")

    evidences = build_evidence_blocks(reranked)
    if not evidences:
        raise HTTPException(
            status_code=404,
            detail="Unable to assemble evidence blocks from retrieved chunks.",
        )

    try:
        answer = answer_generator.generate(payload.question, evidences)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Answer generation failed.") from exc

    return QAResponse(answer=answer, evidences=evidences)
