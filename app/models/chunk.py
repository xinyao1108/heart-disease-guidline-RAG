"""Chunk-level models used for indexing and retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ChunkMeta(BaseModel):
    """Metadata describing the origin of a chunk."""

    guideline_id: str
    guideline_title: str
    year: Optional[int] = None
    organization: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page_range: Optional[Tuple[int, int]] = None
    lang: str = "en"
    rec_class_list: List[str] = Field(default_factory=list)
    loe_list: List[str] = Field(default_factory=list)


class Chunk(ChunkMeta):
    """A chunk ready for indexing."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
