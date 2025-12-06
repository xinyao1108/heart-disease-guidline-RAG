"""Convert parsed paragraphs into retrieval-ready chunks."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from app.config import settings
from app.models.chunk import Chunk
from app.models.document import Paragraph
from app.utils.tokenization import count_tokens, get_cl100k_encoding

logger = logging.getLogger(__name__)

REC_CLASS_PATTERN = re.compile(r"(Class\s+(I{1,3}|IV|V|IIa|IIb))", re.IGNORECASE)
LOE_PATTERN = re.compile(r"(Level\s+(A|B|C))", re.IGNORECASE)

encoding = get_cl100k_encoding("building English guideline chunks")


def read_paragraphs(path: Path) -> Iterator[Paragraph]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield Paragraph(**json.loads(line))


def _normalize_label(label: str) -> str:
    cleaned = re.sub(r"\s+", " ", label.strip())
    parts = cleaned.split(" ", 1)
    if len(parts) == 1:
        return parts[0].capitalize()
    head, tail = parts
    tail_norm = " ".join(part.upper() if part.isalpha() else part for part in tail.split())
    return f"{head.capitalize()} {tail_norm}".strip()


def extract_recommendations(text: str) -> Tuple[List[str], List[str]]:
    rec_classes = sorted({_normalize_label(match[0]) for match in REC_CLASS_PATTERN.findall(text)})
    loe_matches = sorted({_normalize_label(match[0]) for match in LOE_PATTERN.findall(text)})
    return rec_classes, loe_matches


def build_chunk_id(guideline_id: str, section_id: Optional[str], counter: int) -> str:
    section_part = section_id or "section"
    return f"{guideline_id}-{section_part}-{counter:04d}"


def build_chunk(paragraphs: List[Paragraph], counter: int) -> Chunk:
    first = paragraphs[0]
    text = "\n\n".join(p.text for p in paragraphs)
    page_numbers = [p.page for p in paragraphs if p.page]
    page_range = (min(page_numbers), max(page_numbers)) if page_numbers else None
    rec_classes, loe_list = extract_recommendations(text)
    chunk_id = build_chunk_id(first.guideline_id, first.section_id, counter)
    return Chunk(
        chunk_id=chunk_id,
        guideline_id=first.guideline_id,
        guideline_title=first.guideline_title,
        year=first.year,
        organization=first.organization,
        section_id=first.section_id,
        section_title=first.section_title,
        page_range=page_range,
        lang=first.lang,
        text=text,
        rec_class_list=rec_classes,
        loe_list=loe_list,
        metadata={
            "paragraph_ids": [p.order for p in paragraphs],
            "paragraph_count": len(paragraphs),
        },
    )


def chunk_section(paragraphs: List[Paragraph], counter_start: int) -> Iterator[Chunk]:
    buffer: List[Paragraph] = []
    current_tokens = 0
    counter = counter_start

    def flush_buffer() -> Optional[Chunk]:
        nonlocal buffer, current_tokens, counter
        if not buffer:
            return None
        counter += 1
        chunk = build_chunk(buffer, counter)
        overlap = settings.chunk_overlap
        buffer = buffer[-overlap:] if overlap else []
        current_tokens = sum(count_tokens(p.text, encoding) for p in buffer)
        return chunk

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph.text, encoding)
        if (
            buffer
            and current_tokens + paragraph_tokens > settings.chunk_max_tokens
        ):
            chunk = flush_buffer()
            if chunk:
                yield chunk
        buffer.append(paragraph)
        current_tokens += paragraph_tokens
        if current_tokens >= settings.chunk_target_tokens:
            chunk = flush_buffer()
            if chunk:
                yield chunk
    chunk = flush_buffer()
    if chunk:
        yield chunk


def chunk_paragraphs(paragraphs: Iterator[Paragraph]) -> Iterator[Chunk]:
    counter_by_guideline: Dict[str, int] = defaultdict(int)
    current_key: Optional[Tuple[str, Optional[str]]] = None
    section_buffer: List[Paragraph] = []

    def emit_section() -> Iterator[Chunk]:
        nonlocal section_buffer
        if not section_buffer:
            return iter([])
        guideline_id = section_buffer[0].guideline_id
        counter = counter_by_guideline[guideline_id]
        chunks = list(chunk_section(section_buffer, counter))
        counter_by_guideline[guideline_id] += len(chunks)
        section_buffer = []
        return iter(chunks)

    for paragraph in paragraphs:
        key = (paragraph.guideline_id, paragraph.section_id)
        if current_key is None:
            current_key = key
        if key != current_key:
            for chunk in emit_section():
                yield chunk
            current_key = key
        section_buffer.append(paragraph)

    for chunk in emit_section():
        yield chunk


def write_chunks(chunks: Iterable[Chunk], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for count, chunk in enumerate(chunks, start=1):
            handle.write(json.dumps(chunk.model_dump()) + "\n")
    return count


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    paragraphs_path = settings.parsed_docs_path_obj
    if not paragraphs_path.exists():
        logger.error("Parsed documents not found at %s", paragraphs_path)
        return
    chunks = chunk_paragraphs(read_paragraphs(paragraphs_path))
    total = write_chunks(chunks, settings.chunks_path_obj)
    logger.info("Wrote %s chunk rows to %s", total, settings.chunks_path)


if __name__ == "__main__":
    main()
