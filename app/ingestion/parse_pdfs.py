"""Parse guideline PDFs into structured paragraphs."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import fitz

from app.config import settings
from app.ingestion.scan_guidelines import discover_guidelines
from app.models.document import DocumentMeta, Paragraph, Section

logger = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(r"^(?P<num>\d+(?:\.\d+)*)(?:\s+)(?P<title>.+)$")
UPPER_SECTION_PATTERN = re.compile(r"^[A-Z0-9 ,;:/()-]{6,}$")


def normalize_block_text(text: str) -> str:
    parts = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(parts)


def detect_section(text: str) -> Optional[Section]:
    """Return a Section object if the text looks like a heading."""
    stripped = text.strip()
    if not stripped:
        return None
    match = SECTION_PATTERN.match(stripped)
    if match:
        return Section(
            guideline_id="",
            section_id=match.group("num"),
            section_title=match.group("title").strip(),
        )
    if len(stripped.split()) <= 10 and UPPER_SECTION_PATTERN.match(stripped):
        return Section(guideline_id="", section_title=stripped.title())
    return None


def iter_page_paragraphs(page: fitz.Page) -> Iterable[str]:
    """Yield cleaned text blocks from a PDF page."""
    blocks = page.get_text("blocks")
    for block in sorted(blocks, key=lambda b: (b[1], b[0])):
        text = normalize_block_text(block[4])
        if text:
            yield text


def parse_document(meta: DocumentMeta) -> List[Paragraph]:
    """Parse a single PDF into paragraphs."""
    doc = fitz.open(meta.source_path)
    paragraphs: List[Paragraph] = []
    section_counter = 0
    current_section = Section(
        guideline_id=meta.guideline_id,
        section_id=None,
        section_title=None,
    )
    order = 0

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        for block_text in iter_page_paragraphs(doc[page_index]):
            detected = detect_section(block_text)
            if detected:
                section_counter += 1
                current_section = Section(
                    guideline_id=meta.guideline_id,
                    section_id=detected.section_id or f"{section_counter}",
                    section_title=detected.section_title,
                    page_start=page_number,
                    page_end=page_number,
                )
                continue

            order += 1
            paragraphs.append(
                Paragraph(
                    guideline_id=meta.guideline_id,
                    guideline_title=meta.title,
                    year=meta.year,
                    organization=meta.organization,
                    section_id=current_section.section_id,
                    section_title=current_section.section_title,
                    page=page_number,
                    order=order,
                    text=block_text,
                )
            )
    doc.close()
    logger.debug("Parsed %s paragraphs from %s", len(paragraphs), meta.source_path)
    return paragraphs


def parse_all_guidelines() -> None:
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting PDF parsing from %s", settings.guideline_root)
    metas = discover_guidelines()
    if not metas:
        logger.error("No guideline metadata found. Update GUIDELINE_ROOT and retry.")
        return

    output_path = settings.parsed_docs_path_obj
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for meta in metas:
            for paragraph in parse_document(meta):
                total += 1
                handle.write(json.dumps(paragraph.model_dump()) + "\n")
    logger.info("Wrote %s paragraph rows to %s", total, output_path)


def main() -> None:
    parse_all_guidelines()


if __name__ == "__main__":
    main()
