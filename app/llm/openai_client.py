"""Thin wrapper around the OpenAI Responses API."""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from app.config import settings


class OpenAIChatClient:
    """Lazily initializes the OpenAI Python SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured in the environment.")
        self.model = model or settings.openai_model_chat
        self.client = OpenAI(api_key=api_key, base_url=base_url or settings.openai_base_url)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        chunks: list[str] = []
        for item in response.output or []:
            for content in getattr(item, "content", []):
                content_type = getattr(content, "type", None)
                content_text = getattr(content, "text", None)
                if isinstance(content, dict):
                    content_type = content.get("type", content_type)
                    content_text = content.get("text", content_text)
                if content_type in {"output_text", "text"} and content_text:
                    chunks.append(str(content_text))
        return "\n".join(part.strip() for part in chunks if part).strip()
