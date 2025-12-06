"""Utilities for assembling evidence blocks from retrieved chunks."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.models.retrieval import EvidenceBlock, RetrievedChunk
from app.utils.tokenization import count_tokens, get_cl100k_encoding

logger = logging.getLogger(__name__)


encoding = get_cl100k_encoding("assembling evidence blocks")


def _merge_range(existing: Optional[Tuple[int, int]], incoming: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not existing:
        return incoming
    if not incoming:
        return existing
    return (min(existing[0], incoming[0]), max(existing[1], incoming[1]))


def build_evidence_blocks(
    chunks: List[RetrievedChunk],
    max_blocks: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> List[EvidenceBlock]:
    """Group retrieved chunks into prompt-friendly evidence blocks."""
    if max_blocks is None:
        max_blocks = settings.max_evidence_blocks
    if max_tokens is None:
        max_tokens = settings.max_evidence_tokens

    grouped: Dict[Tuple[str, Optional[str]], EvidenceBlock] = {}
    ordered: List[EvidenceBlock] = []

    for chunk in chunks:
        key = (chunk.guideline_id, chunk.section_id or chunk.chunk_id)
        block = grouped.get(key)
        if not block:
            block = EvidenceBlock(
                id="",
                doc_id=chunk.guideline_id,
                guideline_id=chunk.guideline_id,
                guideline_title=chunk.guideline_title,
                year=chunk.year,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                page_range=chunk.page_range,
                text=chunk.text,
                rec_class_list=chunk.rec_class_list,
                loe_list=chunk.loe_list,
            )
            grouped[key] = block
            ordered.append(block)
        else:
            block.text = f"{block.text}\n\n{chunk.text}"
            block.page_range = _merge_range(block.page_range, chunk.page_range)
            block.rec_class_list = sorted({*block.rec_class_list, *chunk.rec_class_list})
            block.loe_list = sorted({*block.loe_list, *chunk.loe_list})

    selected: List[EvidenceBlock] = []
    tokens_left = max_tokens
    for idx, block in enumerate(ordered, start=1):
        if len(selected) >= max_blocks:
            break
        block_tokens = count_tokens(block.text, encoding)
        if block_tokens > tokens_left and selected:
            break
        block.id = f"Doc {idx}"
        tokens_left = max(tokens_left - block_tokens, 0)
        selected.append(block)
    return selected
