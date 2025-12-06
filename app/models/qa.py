"""Request/response models for the public API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .retrieval import EvidenceBlock


class QARequest(BaseModel):
    """Incoming question payload."""

    question: str = Field(..., min_length=3)


class QAResponse(BaseModel):
    """Answer returned to the caller."""

    answer: str
    evidences: List[EvidenceBlock]
