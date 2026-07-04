"""
Image understanding.

Two complementary signals are extracted per image so it becomes searchable:
  1. OCR text (pytesseract) — catches literal text in screenshots, scanned
     forms, charts with axis labels, etc.
  2. Vision-model caption (Claude, multimodal) — catches semantic content
     ("bar chart showing declining Q3 margins in EMEA") that OCR can't.

Both get concatenated into the chunk's searchable text and stored
separately in `extracted_images` for citation/display purposes.
"""

import base64
import pytesseract
from PIL import Image
from anthropic import Anthropic

from app.config import settings

_client = Anthropic(api_key=settings.anthropic_api_key)


def ocr_image(image_path: str) -> str:
    try:
        return pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception:
        return ""


def caption_image(image_path: str) -> str:
    """Ask Claude to describe the image for retrieval purposes: what it
    depicts, key numbers/labels, and why it might matter in an enterprise
    document (chart, diagram, screenshot, signature, etc.)."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    media_type = "image/png" if image_path.lower().endswith("png") else "image/jpeg"

    response = _client.messages.create(
        model=settings.llm_model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": (
                    "Describe this image for a document search index. Include any "
                    "visible numbers, labels, chart type, or diagram structure. "
                    "Be factual and concise (2-4 sentences)."
                )},
            ],
        }],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def process_image(image_path: str) -> dict:
    ocr_text = ocr_image(image_path)
    description = caption_image(image_path)
    searchable_text = f"{description}\n{ocr_text}".strip()
    return {
        "description": description,
        "ocr_text": ocr_text,
        "searchable_text": searchable_text,
    }
