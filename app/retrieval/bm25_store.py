"""Tantivy-based sparse retriever."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import tantivy

from app.config import settings
from app.ingestion.index_bm25 import build_schema
from app.models.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Store:
    """Wrapper around a Tantivy index."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or settings.bm25_index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"BM25 index directory {self.index_dir} does not exist. Run index_bm25 first."
            )
        schema = build_schema()
        self.index = tantivy.Index(schema, path=str(self.index_dir), reuse=True)
        self.searcher = self.index.searcher()
        self.fields = {
            "chunk_id": schema.get_field("chunk_id"),
            "guideline_id": schema.get_field("guideline_id"),
            "guideline_title": schema.get_field("guideline_title"),
            "section_id": schema.get_field("section_id"),
            "section_title": schema.get_field("section_title"),
            "organization": schema.get_field("organization"),
            "year": schema.get_field("year"),
            "text": schema.get_field("text"),
            "lang": schema.get_field("lang"),
            "page_range": schema.get_field("page_range"),
            "rec_classes": schema.get_field("rec_classes"),
            "loe_list": schema.get_field("loe_list"),
        }

    def _parse_query(self, query_text: str) -> tantivy.Query:
        query, errors = self.index.parse_query_lenient(
            query_text,
            default_field_names=[
                self.fields["text"],
                self.fields["section_title"],
                self.fields["guideline_title"],
            ],
            field_boosts={
                self.fields["section_title"]: 2.0,
                self.fields["guideline_title"]: 1.5,
                self.fields["text"]: 1.0,
            },
        )
        if errors:
            logger.debug("Tantivy lenient parse warnings: %s", errors)
        return query

    def search(self, query_text: str, top_k: int = 32) -> List[RetrievedChunk]:
        query = self._parse_query(query_text)
        result = self.searcher.search(query, limit=top_k)
        retrieved: List[RetrievedChunk] = []
        for score, doc_addr in result.hits:
            stored = self.searcher.doc(doc_addr)
            page_range = stored.get("page_range")
            parsed_page_range = None
            if page_range:
                parts = page_range.split("-")
                if len(parts) == 2:
                    parsed_page_range = (int(parts[0]), int(parts[1]))
            rec_classes_raw = stored.get("rec_classes", [""])[0]
            loe_raw = stored.get("loe_list", [""])[0]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=stored["chunk_id"][0],
                    guideline_id=stored["guideline_id"][0],
                    guideline_title=stored["guideline_title"][0],
                    section_id=stored["section_id"][0] or None,
                    section_title=stored["section_title"][0] or None,
                    organization=stored["organization"][0] or None,
                    year=int(stored["year"][0]) if stored["year"][0] else None,
                    text=stored["text"][0],
                    lang=stored["lang"][0],
                    page_range=parsed_page_range,
                    rec_class_list=[
                        item.strip() for item in rec_classes_raw.split(";") if item.strip()
                    ],
                    loe_list=[item.strip() for item in loe_raw.split(";") if item.strip()],
                    sparse_score=float(score),
                    metadata={},
                )
            )
        return retrieved
