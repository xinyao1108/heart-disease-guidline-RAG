"""Glue module that turns evidence blocks into an answer via OpenAI."""

from __future__ import annotations

from typing import Iterable, List

from app.config import settings
from app.llm.openai_client import OpenAIChatClient
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.models.retrieval import EvidenceBlock


class AnswerGenerator:
    """Generates guideline-grounded answers."""

    def __init__(self, client: OpenAIChatClient | None = None) -> None:
        self.client = client or OpenAIChatClient()

    def generate(self, question: str, evidences: Iterable[EvidenceBlock]) -> str:
        evidence_list = list(evidences)
        prompt = build_user_prompt(question, evidence_list)
        raw_answer = self.client.complete(SYSTEM_PROMPT, prompt)
        disclaimer = settings.medical_disclaimer.strip()
        if disclaimer.lower() not in raw_answer.lower():
            raw_answer = f"{raw_answer.rstrip()}\n\n{disclaimer}."
        return raw_answer.strip()
