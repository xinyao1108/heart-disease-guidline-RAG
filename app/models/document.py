"""Document-level data models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DocumentMeta(BaseModel):
    """Metadata captured for each guideline PDF."""

    guideline_id: str
    title: str
    year: Optional[int] = None
    organization: Optional[str] = None
    language: str = "en"
    source_path: str


class Section(BaseModel):
    """Section-level metadata extracted from the PDF."""

    guideline_id: str
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class Paragraph(BaseModel):
    """Paragraph-level text representation."""

    guideline_id: str
    guideline_title: str
    year: Optional[int] = None
    organization: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page: Optional[int] = None
    order: int
    lang: str = "en"
    text: str
