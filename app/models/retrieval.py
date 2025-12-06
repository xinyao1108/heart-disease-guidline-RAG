"""Retrieval request/response models."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from .chunk import Chunk


class RetrievedChunk(Chunk):
    """Chunk extended with retrieval scores."""

    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class EvidenceBlock(BaseModel):
    """Evidence block grouped for prompting."""

    id: str
    doc_id: str
    guideline_id: str
    guideline_title: str
    year: Optional[int] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page_range: Optional[Tuple[int, int]] = None
    text: str
    rec_class_list: List[str] = Field(default_factory=list)
    loe_list: List[str] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    """Payload describing a retrieval job."""

    question: str
    top_k_sparse: int = 32
    top_k_dense: int = 32
    top_k_final: int = 20


class RetrievalResponse(BaseModel):
    """Response returned by the retrieval controller."""

    question: str
    evidences: List[EvidenceBlock]
