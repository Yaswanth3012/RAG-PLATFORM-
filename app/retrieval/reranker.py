"""
Optional cross-encoder reranking pass.

RRF fusion gives a good candidate set fast, but it never actually looks at
query and chunk text together — it only combines two independent rankings.
A cross-encoder scores (query, chunk) pairs jointly and materially improves
precision at the very top of the list, which matters most since only the
top ~5-8 chunks get sent to the LLM.

Kept as a separate, swappable stage: cheap to disable for latency-sensitive
paths, cheap to swap models (e.g. bge-reranker-v2 vs a hosted API).
"""

from sentence_transformers import CrossEncoder

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _model


def rerank(query: str, chunks: list, top_n: int) -> list:
    """chunks: list[RetrievedChunk] from hybrid_search.py. Returns the
    reordered, truncated list."""
    if not chunks:
        return chunks
    model = _get_model()
    pairs = [(query, c.payload.get("text_preview", "")) for c in chunks]
    scores = model.predict(pairs)
    ranked = [c for c, _ in sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)]
    return ranked[:top_n]
