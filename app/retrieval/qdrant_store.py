"""
Qdrant wrapper: collection setup, upsert, and dense-vector search with
metadata filtering. Sparse/keyword retrieval lives in sparse_search.py;
hybrid_search.py fuses the two.
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings

client = QdrantClient(url=settings.qdrant_url)


def ensure_collection():
    """Create the collection with a dense vector plus payload indexes for
    every field we filter on. Payload indexes are what let metadata
    filtering stay fast at 100k+ documents / millions of chunks — without
    them Qdrant falls back to a full payload scan per filtered query."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=settings.embedding_dim, distance=qm.Distance.COSINE),
        # Enables server-side sparse vectors too, if you want Qdrant-native
        # hybrid search instead of the in-process BM25 fallback:
        sparse_vectors_config={"text-sparse": qm.SparseVectorParams()},
    )

    for field_name, schema in [
        ("access_tags", qm.PayloadSchemaType.KEYWORD),
        ("department", qm.PayloadSchemaType.KEYWORD),
        ("classification", qm.PayloadSchemaType.KEYWORD),
        ("content_type", qm.PayloadSchemaType.KEYWORD),
        ("document_id", qm.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=settings.qdrant_collection, field_name=field_name, field_schema=schema
        )


def upsert_chunk_vectors(
    chunk_ids: list[str], vectors: list[list[float]], texts: list[str], payloads: list[dict]
):
    points = [
        qm.PointStruct(id=chunk_ids[i], vector=vectors[i], payload={**payloads[i], "text_preview": texts[i][:500]})
        for i in range(len(chunk_ids))
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


def dense_search(query_vector: list[float], top_k: int, qdrant_filter=None) -> list[dict]:
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points
    return [{"chunk_id": str(r.id), "score": r.score, "payload": r.payload} for r in results]
