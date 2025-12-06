"""Helpers for loading tiktoken encodings with operator-controlled fallback."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import tiktoken

from app.config import settings

logger = logging.getLogger(__name__)


def _should_fallback(context: str, reason: Exception) -> bool:
    message = (
        f"Failed to load tiktoken 'cl100k_base' while {context}. "
        f"Reason: {reason}"
    )
    if settings.allow_tiktoken_fallback:
        logger.warning(
            "%s. Proceeding with whitespace token approximation because ALLOW_TIKTOKEN_FALLBACK=1.",
            message,
        )
        return True

    if sys.stdin.isatty():
        prompt = (
            f"{message}.\n"
            "Type 'fallback' to continue with approximate whitespace token counts, "
            "or press Enter to abort: "
        )
        choice = input(prompt)
        if choice.strip().lower() in {"fallback", "f", "y", "yes"}:
            logger.warning("Operator approved whitespace token fallback for %s.", context)
            return True
        raise RuntimeError("Operator rejected tiktoken fallback; aborting ingestion.")

    raise RuntimeError(
        f"{message}. Rerun with ALLOW_TIKTOKEN_FALLBACK=1 to allow whitespace fallback."
    )


def get_cl100k_encoding(context: str) -> Optional[tiktoken.Encoding]:
    """Try to load the OpenAI tokenizer with an optional operator-approved fallback."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        if _should_fallback(context, exc):
            return None
        raise


def count_tokens(text: str, encoding: Optional[tiktoken.Encoding]) -> int:
    """Count tokens using tiktoken if available, otherwise whitespace approximation."""
    if encoding:
        return len(encoding.encode(text))
    return len(text.split())
