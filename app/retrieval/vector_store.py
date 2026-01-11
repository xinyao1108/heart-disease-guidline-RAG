"""Qdrant-based dense vector store."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.models.retrieval import RetrievedChunk


class VectorStore:
    """Wrapper around Qdrant search."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url, api_key=api_key or settings.qdrant_api_key)
        self.collection = collection or settings.qdrant_collection

    def search(self, query_vector: Sequence[float], top_k: int = 32, lang: str = "en") -> List[RetrievedChunk]:
        filters = None
        if lang:
            filters = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="lang",
                        match=qmodels.MatchValue(value=lang),
                    )
                ]
            )
        results = None
        search_fn = getattr(self.client, "search", None)
        if search_fn is not None:
            results = search_fn(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=None,
                query_filter=filters,
            )
        else:
            search_points_fn = getattr(self.client, "search_points", None)
            if search_points_fn is not None:
                results = search_points_fn(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=None,
                    query_filter=filters,
                )
            else:
                http_search = getattr(getattr(self.client, "http", None), "search_api", None)
                if http_search and hasattr(http_search, "search_points"):
                    response = http_search.search_points(
                        collection_name=self.collection,
                        search_request=qmodels.SearchRequest(
                            vector=query_vector,
                            filter=filters,
                            limit=top_k,
                            with_payload=True,
                        ),
                    )
                    results = response.result or []
                else:
                    raise AttributeError("Qdrant client does not support search/search_points.")
        retrieved: List[RetrievedChunk] = []
        for point in results:
            payload = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id") or str(point.id),
                    guideline_id=payload.get("guideline_id"),
                    guideline_title=payload.get("guideline_title"),
                    section_id=payload.get("section_id"),
                    section_title=payload.get("section_title"),
                    organization=payload.get("organization"),
                    year=payload.get("year"),
                    text=payload.get("text", ""),
                    lang=payload.get("lang", lang),
                    page_range=tuple(payload.get("page_range", [])) if payload.get("page_range") else None,
                    rec_class_list=payload.get("rec_class_list", []),
                    loe_list=payload.get("loe_list", []),
                    metadata=payload.get("metadata", {}),
                    dense_score=float(point.score) if point.score is not None else None,
                )
            )
        return retrieved
