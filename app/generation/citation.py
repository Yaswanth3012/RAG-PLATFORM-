"""
Citation handling.

We give each retrieved chunk a stable inline marker like [1], [2] before
sending context to the LLM, instruct it to cite using those markers, and
then resolve markers back to (document, page, chunk) after generation so
the UI can render clickable source links — never trusting the LLM to
report sources from memory.
"""

from dataclasses import dataclass


@dataclass
class Source:
    marker: int
    document_id: str
    document_title: str
    page_number: int | None
    chunk_id: str
    content_type: str
    snippet: str


def build_context_block(chunks: list) -> tuple[str, list[Source]]:
    """chunks: list[RetrievedChunk]. Returns the prompt-ready context string
    and the ordered Source list for later citation resolution."""
    lines = []
    sources = []
    for i, c in enumerate(chunks, start=1):
        p = c.payload
        lines.append(
            f"[{i}] (doc: {p.get('title')}, page: {p.get('page_number')}, type: {p.get('content_type')})\n"
            f"{p.get('text_preview', '')}"
        )
        sources.append(Source(
            marker=i,
            document_id=p.get("document_id"),
            document_title=p.get("title"),
            page_number=p.get("page_number"),
            chunk_id=c.chunk_id,
            content_type=p.get("content_type", "text"),
            snippet=p.get("text_preview", "")[:300],
        ))
    return "\n\n".join(lines), sources


def resolve_cited_sources(answer_text: str, sources: list[Source]) -> list[Source]:
    """Extract which [N] markers actually appear in the model's answer,
    so the UI only shows sources that were used — not the full retrieved set."""
    used_markers = set()
    for s in sources:
        if f"[{s.marker}]" in answer_text:
            used_markers.add(s.marker)
    return [s for s in sources if s.marker in used_markers]
