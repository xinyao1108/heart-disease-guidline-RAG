"""Scan the guideline directory and collect metadata."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

from app.config import settings
from app.models.document import DocumentMeta

logger = logging.getLogger(__name__)

CHINESE_KEYWORD = "心内科指南"
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def normalize_guideline_id(path: Path) -> str:
    """Generate a deterministic identifier from the file name."""
    return path.stem.lower().replace(" ", "-")


def guess_year(name: str) -> Optional[int]:
    match = YEAR_PATTERN.search(name)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


def guess_org(name: str) -> Optional[str]:
    name_lower = name.lower()
    orgs: List[str] = []
    for token, label in {
        "aha": "AHA",
        "acc": "ACC",
        "hrs": "HRS",
        "esc": "ESC",
        "hfsa": "HFSA",
        "scai": "SCAI",
    }.items():
        if token in name_lower:
            orgs.append(label)
    return "/".join(orgs) if orgs else None


def discover_guidelines(root: Optional[Path] = None) -> List[DocumentMeta]:
    """Return metadata for every allowed guideline PDF."""
    root_path = root or settings.guideline_root_path
    if not root_path.exists():
        logger.warning("Guideline root %s does not exist", root_path)
        return []

    pdfs = sorted(root_path.rglob("*.pdf"))
    metas: List[DocumentMeta] = []
    for pdf in pdfs:
        if CHINESE_KEYWORD in pdf.parts:
            continue
        guideline_id = normalize_guideline_id(pdf)
        year = guess_year(pdf.name)
        title = pdf.stem.replace("-", " ").replace("_", " ").title()
        metas.append(
            DocumentMeta(
                guideline_id=guideline_id,
                title=title,
                year=year,
                organization=guess_org(pdf.stem),
                source_path=str(pdf.resolve()),
            )
        )
    logger.info("Discovered %s English guideline PDFs", len(metas))
    return metas


def export_metadata(metas: Iterable[DocumentMeta], output_path: Path) -> None:
    """Persist metadata as JSON lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for count, meta in enumerate(metas, start=1):
            handle.write(json.dumps(meta.model_dump()) + "\n")
    logger.info("Wrote %s guideline metadata rows to %s", count, output_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=settings.log_level)
    metas = discover_guidelines()
    if not metas:
        logger.error("No guidelines found at %s", settings.guideline_root)
        return
    output = settings.parsed_docs_path_obj.with_name("english_guidelines_meta.jsonl")
    export_metadata(metas, output)


if __name__ == "__main__":
    main()
