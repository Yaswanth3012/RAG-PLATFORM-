"""
Format-aware loading. `unstructured` gives us a single interface across
PDF/DOCX/HTML/images but we post-process its output into typed elements
(text / table / image) so downstream chunking and embedding can treat
each modality correctly.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from unstructured.partition.auto import partition
from unstructured.documents.elements import Table, Image as UnstructuredImage


@dataclass
class RawElement:
    text: str
    element_type: str  # "text" | "table" | "image"
    page_number: int | None
    metadata: dict = field(default_factory=dict)


def compute_checksum(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_document(file_path: str) -> list[RawElement]:
    """Partition a document into typed elements using `unstructured`.

    Table elements retain their HTML/text representation for later
    structured extraction (see table_extraction.py). Image elements are
    handed off to image_understanding.py for captioning + OCR.
    """
    elements = partition(
        filename=file_path,
        strategy="hi_res",          # needed for layout-aware table/image detection
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )

    raw_elements: list[RawElement] = []
    for el in elements:
        page_number = getattr(el.metadata, "page_number", None)

        if isinstance(el, Table):
            raw_elements.append(RawElement(
                text=str(el),
                element_type="table",
                page_number=page_number,
                metadata={"text_as_html": getattr(el.metadata, "text_as_html", None)},
            ))
        elif isinstance(el, UnstructuredImage):
            raw_elements.append(RawElement(
                text=str(el),
                element_type="image",
                page_number=page_number,
                metadata={"image_path": getattr(el.metadata, "image_path", None)},
            ))
        else:
            raw_elements.append(RawElement(
                text=str(el),
                element_type="text",
                page_number=page_number,
            ))

    return raw_elements


def infer_doc_type(file_path: str) -> str:
    return Path(file_path).suffix.lstrip(".").lower()
