"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings."""

    openai_api_key: Optional[str] = Field(
        default=None, description="Secret key for OpenAI APIs."
    )
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_chat: str = "gpt-4.1-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "guideline_chunks_en"

    guideline_root: str = "data/raw/english_guidelines"
    parsed_docs_path: str = "data/parsed/english_docs.jsonl"
    chunks_path: str = "data/chunks/english_chunks.jsonl"
    bm25_index_dir: str = "data/bm25_index"

    chunk_target_tokens: int = 320
    chunk_max_tokens: int = 420
    chunk_overlap: int = 1
    max_evidence_blocks: int = 6
    max_evidence_tokens: int = 3000

    log_level: str = "INFO"
    medical_disclaimer: str = (
        "This information is for educational purposes only and is not a substitute "
        "for professional medical advice. Always consult qualified clinicians"
    )
    allow_tiktoken_fallback: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def guideline_root_path(self) -> Path:
        return Path(self.guideline_root)

    @property
    def parsed_docs_path_obj(self) -> Path:
        return Path(self.parsed_docs_path)

    @property
    def chunks_path_obj(self) -> Path:
        return Path(self.chunks_path)

    @property
    def bm25_index_path_obj(self) -> Path:
        return Path(self.bm25_index_dir)


settings = Settings()
