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

    TEXT_FIELDS = ["text", "section_title", "guideline_title"]

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or settings.bm25_index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"BM25 index directory {self.index_dir} does not exist. Run index_bm25 first."
            )
        schema = build_schema()
        self.index = tantivy.Index(schema, path=str(self.index_dir), reuse=True)
        self.searcher = self.index.searcher()

    def _parse_query(self, query_text: str) -> tantivy.Query:
        escaped = query_text.replace('"', " ").replace("\\", " ").strip()
        if not escaped:
            escaped = "*"
        field_parts = [
            f'section_title:({escaped})^2.0',
            f'guideline_title:({escaped})^1.5',
            f'text:({escaped})^1.0',
        ]
        combined = " OR ".join(field_parts)
        query, errors = self.index.parse_query_lenient(combined)
        if errors:
            logger.debug("Tantivy lenient parse warnings: %s", errors)
        return query

    def search(self, query_text: str, top_k: int = 32) -> List[RetrievedChunk]:
        query = self._parse_query(query_text)
        result = self.searcher.search(query, limit=top_k)
        retrieved: List[RetrievedChunk] = []
        for score, doc_addr in result.hits:
            stored = self.searcher.doc(doc_addr)
            if hasattr(stored, "to_dict"):
                stored_fields = stored.to_dict()

                def field_values(key: str, default=None):
                    return stored_fields.get(key, default)

            else:

                def field_values(key: str, default=None):
                    value = stored.get(key)
                    return value if value is not None else default

            page_range = field_values("page_range")
            parsed_page_range = None
            if page_range:
                raw_range = page_range[0] if isinstance(page_range, list) else page_range
                parts = raw_range.split("-")
                if len(parts) == 2:
                    parsed_page_range = (int(parts[0]), int(parts[1]))
            rec_classes_raw = (field_values("rec_classes") or [""])[0]
            loe_raw = (field_values("loe_list") or [""])[0]
            year_value = (field_values("year") or [""])[0]
            section_id_val = (field_values("section_id") or [""])[0]
            section_title_val = (field_values("section_title") or [""])[0]
            org_val = (field_values("organization") or [""])[0]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=field_values("chunk_id", [""])[0],
                    guideline_id=field_values("guideline_id", [""])[0],
                    guideline_title=field_values("guideline_title", [""])[0],
                    section_id=section_id_val or None,
                    section_title=section_title_val or None,
                    organization=org_val or None,
                    year=int(year_value) if year_value else None,
                    text=field_values("text", [""])[0],
                    lang=field_values("lang", ["en"])[0],
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
