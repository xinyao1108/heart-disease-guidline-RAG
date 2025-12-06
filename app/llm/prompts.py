"""Prompt templates for the question answering stage."""

from __future__ import annotations

from typing import Iterable, List

from app.models.retrieval import EvidenceBlock

SYSTEM_PROMPT = """You are MedAgenticSystem, a clinical guideline assistant for cardiovascular care.
Use only the provided evidence blocks to answer clinician questions.
Summaries must be concise, actionable, and reference the supporting evidence using [Doc #] notation.
Highlight relevant patient populations, disease states, classes of recommendation, and levels of evidence whenever possible.
Do not fabricate data. If the evidence is insufficient, say so explicitly."""


def format_evidence_block(block: EvidenceBlock) -> str:
    page_info = ""
    if block.page_range:
        page_info = f" (pages {block.page_range[0]}-{block.page_range[1]})"
    rec_info = ""
    if block.rec_class_list or block.loe_list:
        rec_info = f"\nRecommendations: {', '.join(block.rec_class_list + block.loe_list)}"
    return (
        f"[{block.id}] {block.guideline_title} ({block.year}) - {block.section_title or 'General'}{page_info}\n"
        f"{block.text.strip()}{rec_info}"
    )


def build_user_prompt(question: str, evidences: Iterable[EvidenceBlock]) -> str:
    evidence_sections = "\n\n".join(format_evidence_block(block) for block in evidences)
    return f"""Question:
{question.strip()}

Evidence:
{evidence_sections}

Instructions:
- Answer in English.
- Support every conclusion with a citation such as [Doc 1].
- Mention recommendation class/level when available.
- End with a brief safety disclaimer."""
