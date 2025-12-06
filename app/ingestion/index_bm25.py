"""Build a Tantivy BM25 index over the chunks."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterable

import tantivy

from app.config import settings
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


def load_chunks(path: Path) -> Iterable[Chunk]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield Chunk(**json.loads(line))


def build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True)
    builder.add_text_field("guideline_id", stored=True)
    builder.add_text_field("guideline_title", stored=True)
    builder.add_text_field("section_id", stored=True)
    builder.add_text_field("section_title", stored=True)
    builder.add_text_field("organization", stored=True)
    builder.add_text_field("year", stored=True)
    builder.add_text_field("text", stored=True)
    builder.add_text_field("lang", stored=True)
    builder.add_text_field("page_range", stored=True)
    builder.add_text_field("rec_classes", stored=True)
    builder.add_text_field("loe_list", stored=True)
    return builder.build()


def prepare_index(schema: tantivy.Schema, index_dir: Path) -> tantivy.Index:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    return tantivy.Index(schema, path=str(index_dir), reuse=False)


def add_chunk(writer: tantivy.IndexWriter, chunk: Chunk) -> None:
    page_range = ""
    if chunk.page_range:
        page_range = f"{chunk.page_range[0]}-{chunk.page_range[1]}"
    doc = {
        "chunk_id": chunk.chunk_id,
        "guideline_id": chunk.guideline_id,
        "guideline_title": chunk.guideline_title,
        "section_id": chunk.section_id or "",
        "section_title": chunk.section_title or "",
        "organization": chunk.organization or "",
        "year": str(chunk.year or ""),
        "text": chunk.text,
        "lang": chunk.lang,
        "page_range": page_range,
        "rec_classes": ";".join(chunk.rec_class_list),
        "loe_list": ";".join(chunk.loe_list),
    }
    writer.add_document(doc)


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    chunks_path = settings.chunks_path_obj
    if not chunks_path.exists():
        logger.error("Chunk file %s does not exist. Run chunking first.", chunks_path)
        return
    schema = build_schema()
    index = prepare_index(schema, settings.bm25_index_path_obj)
    writer = index.writer()
    count = 0
    for chunk in load_chunks(chunks_path):
        count += 1
        add_chunk(writer, chunk)
    writer.commit()
    logger.info("Indexed %s chunks into %s", count, settings.bm25_index_dir)


if __name__ == "__main__":
    main()
