"""
Core RAG generation: retrieval -> prompt assembly -> streamed LLM answer
with citation resolution at the end of the stream.
"""

import time
from typing import AsyncIterator
from anthropic import AsyncAnthropic

from app.config import settings
from app.auth.rbac import AccessContext
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.citation import build_context_block, resolve_cited_sources

_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer ONLY using \
the numbered context sources provided. Every factual claim must include an \
inline citation marker like [1] or [2] pointing to the source it came from. \
If the sources don't contain the answer, say so plainly instead of guessing. \
When a source is a table, cite specific rows/values rather than summarizing vaguely."""


async def answer_query_stream(
    query: str,
    access_ctx: AccessContext,
    metadata_filters: dict | None = None,
    use_reranker: bool = True,
) -> AsyncIterator[dict]:
    """Yields SSE-ready dicts: {"type": "chunk"|"sources"|"done", ...}"""
    start = time.perf_counter()

    retrieved = hybrid_search(query, access_ctx, metadata_filters=metadata_filters)
    if use_reranker and retrieved:
        retrieved = rerank(query, retrieved, top_n=min(6, len(retrieved)))

    if not retrieved:
        yield {"type": "chunk", "text": "I couldn't find any documents you have access to that answer this question."}
        yield {"type": "done", "latency_ms": int((time.perf_counter() - start) * 1000)}
        return

    context_block, sources = build_context_block(retrieved)

    user_message = f"Context sources:\n\n{context_block}\n\nQuestion: {query}"

    answer_parts: list[str] = []
    async with _client.messages.stream(
        model=settings.llm_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            answer_parts.append(text)
            yield {"type": "chunk", "text": text}

    full_answer = "".join(answer_parts)
    used_sources = resolve_cited_sources(full_answer, sources)

    yield {
        "type": "sources",
        "sources": [
            {
                "marker": s.marker,
                "document_id": s.document_id,
                "document_title": s.document_title,
                "page_number": s.page_number,
                "content_type": s.content_type,
                "snippet": s.snippet,
            }
            for s in used_sources
        ],
    }
    yield {"type": "done", "latency_ms": int((time.perf_counter() - start) * 1000)}
