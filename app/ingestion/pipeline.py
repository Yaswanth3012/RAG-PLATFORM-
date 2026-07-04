"""
End-to-end ingestion pipeline.

    file --> load_document --> chunk_elements --> [embed] --> Qdrant (vectors)
                                                            --> Postgres (text, metadata, ACL, tables, images)

Idempotency: we compute a checksum of the file; if a document with the same
checksum already exists and is `indexed`, we skip re-processing (important
at 100k+ document scale where re-ingestion is triggered by scheduled crawls).
"""

import uuid
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Chunk
from app.ingestion.loaders import load_document, compute_checksum, infer_doc_type
from app.ingestion.chunking import chunk_elements
from app.ingestion.table_extraction import extract_table_json
from app.ingestion.image_understanding import process_image
from app.retrieval.qdrant_store import upsert_chunk_vectors
from app.retrieval.embeddings import embed_texts

logger = logging.getLogger("ingestion")


async def ingest_document(
    db: AsyncSession,
    file_path: str,
    department: str,
    classification: str,
    access_tags: list[str],
    ingested_by: str,
    extra_metadata: dict | None = None,
) -> Document:
    checksum = compute_checksum(file_path)

    existing = (await db.execute(
        select(Document).where(Document.checksum == checksum, Document.status == "indexed")
    )).scalar_one_or_none()
    if existing:
        logger.info("Skipping re-ingest of unchanged document %s", file_path)
        return existing

    doc = Document(
        id=uuid.uuid4(),
        source_uri=file_path,
        title=file_path.split("/")[-1],
        doc_type=infer_doc_type(file_path),
        department=department,
        classification=classification,
        access_tags=access_tags,
        checksum=checksum,
        status="processing",
        ingested_by=ingested_by,
        extra_metadata=extra_metadata or {},
    )
    db.add(doc)
    await db.flush()

    try:
        raw_elements = load_document(file_path)

        # Enrich table/image elements with structured extraction before chunking
        for el in raw_elements:
            if el.element_type == "table":
                el.metadata["table_json"] = extract_table_json(
                    el.metadata.get("text_as_html"), file_path, el.page_number
                )
            elif el.element_type == "image" and el.metadata.get("image_path"):
                img_result = process_image(el.metadata["image_path"])
                el.text = img_result["searchable_text"]
                el.metadata.update(img_result)

        chunks = chunk_elements(raw_elements)

        chunk_rows: list[Chunk] = []
        for c in chunks:
            chunk_rows.append(Chunk(
                id=uuid.uuid4(),
                document_id=doc.id,
                chunk_index=c.chunk_index,
                content=c.content,
                content_type=c.content_type,
                page_number=c.page_number,
                token_count=c.token_count,
            ))
        db.add_all(chunk_rows)
        await db.flush()

        # Embed and upsert into Qdrant with ACL + metadata payload for filtering
        vectors = embed_texts([c.content for c in chunk_rows])
        upsert_chunk_vectors(
            chunk_ids=[str(c.id) for c in chunk_rows],
            vectors=vectors,
            texts=[c.content for c in chunk_rows],
            payloads=[{
                "document_id": str(doc.id),
                "chunk_id": str(c.id),
                "access_tags": access_tags,
                "department": department,
                "classification": classification,
                "content_type": c.content_type,
                "page_number": c.page_number,
                "title": doc.title,
                **(extra_metadata or {}),
            } for c in chunk_rows],
        )

        for row, chunk_obj in zip(chunk_rows, chunk_rows):
            chunk_obj.qdrant_point_id = chunk_obj.id  # 1:1 mapping, point id == chunk id

        doc.status = "indexed"
        await db.commit()
        logger.info("Indexed %s (%d chunks)", file_path, len(chunk_rows))
        return doc

    except Exception:
        doc.status = "failed"
        await db.commit()
        logger.exception("Failed to ingest %s", file_path)
        raise
