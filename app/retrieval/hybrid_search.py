"""
Hybrid search = dense (semantic) + sparse (lexical/BM25) retrieval, fused
with Reciprocal Rank Fusion (RRF), then RBAC-filtered, then optionally
re-ranked with a cross-encoder for the final top-N handed to the LLM.

Why RRF instead of a weighted score blend: dense cosine scores and BM25
scores live on different, non-comparable scales, and their distributions
shift per-query. RRF sidesteps calibration entirely by fusing on RANK
rather than raw score, which is far more robust across query types
(a 3-word keyword query behaves very differently than a 30-word question).
"""

from dataclasses import dataclass

from app.config import settings
from app.retrieval.embeddings import embed_query
from app.retrieval.qdrant_store import dense_search
from app.retrieval.sparse_search import sparse_search
from app.auth.rbac import AccessContext, qdrant_access_filter

# Known tag vocabulary — in production this is generated from the
# distinct set of (department, classification) pairs in Postgres and
# cached/refreshed periodically. Kept small and static here for clarity.
KNOWN_TAG_VOCAB = {
    "classification:public", "classification:internal",
    "classification:confidential", "classification:restricted",
    "dept:finance", "dept:hr", "dept:engineering", "dept:legal", "dept:sales",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None


def _rrf_fuse(dense_results: list[dict], sparse_results: list[dict], k: int) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    for rank, r in enumerate(dense_results):
        scores[r["chunk_id"]] = scores.get(r["chunk_id"], 0) + 1.0 / (k + rank + 1)
        payloads[r["chunk_id"]] = r["payload"]
        dense_ranks[r["chunk_id"]] = rank + 1

    for rank, r in enumerate(sparse_results):
        scores[r["chunk_id"]] = scores.get(r["chunk_id"], 0) + 1.0 / (k + rank + 1)
        payloads.setdefault(r["chunk_id"], r["payload"])
        sparse_ranks[r["chunk_id"]] = rank + 1

    fused = [
        RetrievedChunk(
            chunk_id=cid, score=score, payload=payloads[cid],
            dense_rank=dense_ranks.get(cid), sparse_rank=sparse_ranks.get(cid),
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def apply_metadata_filters(chunks: list[RetrievedChunk], filters: dict | None) -> list[RetrievedChunk]:
    """Post-fusion structured filtering: department, date range, doc type,
    custom fields — anything the user specified explicitly in the query UI,
    independent of what RBAC already restricted."""
    if not filters:
        return chunks
    out = []
    for c in chunks:
        ok = True
        for key, value in filters.items():
            payload_value = c.payload.get(key)
            if isinstance(value, list):
                if payload_value not in value:
                    ok = False
                    break
            elif payload_value != value:
                ok = False
                break
        if ok:
            out.append(c)
    return out


def hybrid_search(
    query: str,
    access_ctx: AccessContext,
    metadata_filters: dict | None = None,
    dense_top_k: int = None,
    sparse_top_k: int = None,
    fused_top_k: int = None,
) -> list[RetrievedChunk]:
    dense_top_k = dense_top_k or settings.dense_top_k
    sparse_top_k = sparse_top_k or settings.sparse_top_k
    fused_top_k = fused_top_k or settings.fused_top_k

    query_vector = embed_query(query)
    qfilter = qdrant_access_filter(access_ctx)

    dense_results = dense_search(query_vector, top_k=dense_top_k, qdrant_filter=qfilter)
    sparse_results = sparse_search(
        query, top_k=sparse_top_k,
        allowed_tags=None if access_ctx.is_admin else access_ctx.allowed_tags,
    )

    fused = _rrf_fuse(dense_results, sparse_results, k=settings.rrf_k)
    fused = apply_metadata_filters(fused, metadata_filters)

    return fused[:fused_top_k]
