"""
Sparse/keyword retrieval. Dense embeddings are great at semantic similarity
but weak on exact terms: SKU codes, error codes, acronyms, names — the
things enterprise search actually gets graded on. This module provides
keyword search to fuse with dense results (see hybrid_search.py).

Two backends:
  - elasticsearch: proper inverted index, scales to 100k+ docs, supports
    the same metadata filters as Qdrant via `filter` clauses.
  - bm25 (default/dev fallback): in-process rank_bm25 over a cached corpus.
    Fine for demos and small corpora; NOT what you'd run in production at
    scale — swap RETRIEVAL_BACKEND=elasticsearch for that.
"""

from app.config import settings

_bm25_index = None
_bm25_corpus_ids: list[str] = []
_bm25_corpus_payloads: list[dict] = []


def build_bm25_index(chunk_ids: list[str], texts: list[str], payloads: list[dict]):
    """Populate the in-process BM25 fallback. Call this once at startup by
    pulling all chunk text from Postgres. For real scale, skip this file
    entirely and use the elasticsearch backend below."""
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_payloads
    from rank_bm25 import BM25Okapi

    tokenized = [t.lower().split() for t in texts]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus_ids = chunk_ids
    _bm25_corpus_payloads = payloads


def bm25_search(query: str, top_k: int, allowed_tags: set[str] | None) -> list[dict]:
    if _bm25_index is None:
        return []
    scores = _bm25_index.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked:
        payload = _bm25_corpus_payloads[i]
        if allowed_tags is not None and not set(payload.get("access_tags", [])).issubset(allowed_tags):
            continue
        if scores[i] <= 0:
            continue
        results.append({"chunk_id": _bm25_corpus_ids[i], "score": float(scores[i]), "payload": payload})
        if len(results) >= top_k:
            break
    return results


def elasticsearch_search(query: str, top_k: int, allowed_tags: set[str] | None) -> list[dict]:
    from elasticsearch import Elasticsearch

    es = Elasticsearch(settings.elasticsearch_url)
    es_filter = []
    if allowed_tags is not None:
        es_filter.append({"terms": {"access_tags": list(allowed_tags)}})

    body = {
        "query": {
            "bool": {
                "must": [{"match": {"content": query}}],
                "filter": es_filter,
            }
        },
        "size": top_k,
    }
    resp = es.search(index="rag_chunks", body=body)
    return [
        {"chunk_id": hit["_id"], "score": hit["_score"], "payload": hit["_source"]}
        for hit in resp["hits"]["hits"]
    ]


def sparse_search(query: str, top_k: int, allowed_tags: set[str] | None) -> list[dict]:
    if settings.retrieval_backend == "elasticsearch":
        return elasticsearch_search(query, top_k, allowed_tags)
    return bm25_search(query, top_k, allowed_tags)
