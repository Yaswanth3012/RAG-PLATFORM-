"""
Chunking strategy.

We use semantic-boundary chunking for prose text (split on paragraph/section
breaks, then pack up to a token budget with overlap), but tables and images
are kept as ATOMIC chunks — never split mid-table or mid-caption, since that
destroys the thing a citation is supposed to point to.
"""

from dataclasses import dataclass
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 400
OVERLAP_TOKENS = 60


@dataclass
class Chunk:
    content: str
    content_type: str  # text | table | image_caption
    page_number: int | None
    chunk_index: int
    token_count: int


def _token_len(text: str) -> int:
    return len(ENC.encode(text))


def _pack_text_elements(elements: list[str], pages: list[int | None]) -> list[tuple[str, int | None]]:
    """Greedy packing of consecutive text elements into ~TARGET_TOKENS chunks
    with a sliding overlap, preserving the page number of the chunk's start."""
    packed: list[tuple[str, int | None]] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_start_page = None

    for text, page in zip(elements, pages):
        t = _token_len(text)
        if buf and buf_tokens + t > TARGET_TOKENS:
            packed.append((" ".join(buf), buf_start_page))
            # carry overlap: keep tail tokens worth of text for continuity
            overlap_text = " ".join(buf)[-OVERLAP_TOKENS * 4:]  # rough char proxy
            buf = [overlap_text]
            buf_tokens = _token_len(overlap_text)
            buf_start_page = page
        if not buf:
            buf_start_page = page
        buf.append(text)
        buf_tokens += t

    if buf:
        packed.append((" ".join(buf), buf_start_page))

    return packed


def chunk_elements(raw_elements) -> list[Chunk]:
    """raw_elements: list[RawElement] from loaders.py"""
    chunks: list[Chunk] = []
    idx = 0

    text_buffer: list[str] = []
    page_buffer: list[int | None] = []

    def flush_text_buffer():
        nonlocal idx
        if not text_buffer:
            return
        for content, page in _pack_text_elements(text_buffer, page_buffer):
            chunks.append(Chunk(
                content=content,
                content_type="text",
                page_number=page,
                chunk_index=idx,
                token_count=_token_len(content),
            ))
            idx += 1
        text_buffer.clear()
        page_buffer.clear()

    for el in raw_elements:
        if el.element_type == "text":
            text_buffer.append(el.text)
            page_buffer.append(el.page_number)
        else:
            # table / image — flush any pending prose first, then emit as
            # its own atomic chunk so it's never split.
            flush_text_buffer()
            chunks.append(Chunk(
                content=el.text,
                content_type="table" if el.element_type == "table" else "image_caption",
                page_number=el.page_number,
                chunk_index=idx,
                token_count=_token_len(el.text),
            ))
            idx += 1

    flush_text_buffer()
    return chunks
