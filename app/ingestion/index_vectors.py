"""Build the dense embedding index in Qdrant."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.models.chunk import Chunk
from app.retrieval.embedder import get_bge_m3_embedder

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024
BATCH_SIZE = 8


def load_chunks(path: Path) -> Iterable[Chunk]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield Chunk(**json.loads(line))


def ensure_collection(client: QdrantClient, collection: str) -> None:
    vector_params = qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE)
    if client.collection_exists(collection):
        logger.info("Re-creating existing Qdrant collection %s", collection)
        client.recreate_collection(
            collection_name=collection,
            vectors_config=vector_params,
        )
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=vector_params,
        )


def chunk_batches(items: Iterable[Chunk], batch_size: int) -> Iterable[List[Chunk]]:
    batch: List[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    chunks_path = settings.chunks_path_obj
    if not chunks_path.exists():
        logger.error("Chunk file %s does not exist. Run chunking first.", chunks_path)
        return

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    ensure_collection(client, settings.qdrant_collection)
    embedder = get_bge_m3_embedder()

    count = 0
    for batch in chunk_batches(load_chunks(chunks_path), BATCH_SIZE):
        texts = [chunk.text for chunk in batch]
        embeddings = embedder.encode_corpus(texts)["dense_vecs"]
        points = []
        for chunk, vector in zip(batch, embeddings):
            count += 1
            payload = chunk.model_dump()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        client.upsert(
            collection_name=settings.qdrant_collection,
            wait=True,
            points=points,
        )
    logger.info(
        "Indexed %s chunks into Qdrant collection %s",
        count,
        settings.qdrant_collection,
    )


if __name__ == "__main__":
    main()
