"""Shared helper for loading the BGE-M3 embedding model."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_bge_m3_embedder() -> BGEM3FlagModel:
    """Load the embedding model once per process."""
    return BGEM3FlagModel(MODEL_NAME, use_fp16=False, devices="cpu")


def embed_queries(queries: Iterable[str]) -> List[List[float]]:
    model = get_bge_m3_embedder()
    result = model.encode_queries(list(queries))["dense_vecs"]
    return [vec.tolist() for vec in result]
