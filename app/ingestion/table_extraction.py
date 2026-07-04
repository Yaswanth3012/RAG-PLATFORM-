"""
Structured table extraction.

`unstructured` gives us table text + an HTML rendering. We parse that HTML
into a list-of-rows JSON structure so:
  1. The generation layer can render an accurate markdown table, not a
     garbled flattened string.
  2. Citations can point to "Table 3, row 'Q3 Revenue'" instead of a
     paragraph of run-together numbers.

For scanned/complex PDFs where `unstructured`'s table structure inference
is weak, we fall back to Camelot (lattice/stream) directly on the PDF page.
"""

from io import StringIO
import pandas as pd


def html_table_to_json(html: str) -> list[dict]:
    try:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            return []
        df = dfs[0]
        df.columns = [str(c) for c in df.columns]
        return df.fillna("").to_dict(orient="records")
    except Exception:
        return []


def camelot_fallback(pdf_path: str, page_number: int) -> list[dict]:
    """Used when unstructured's HTML table parse fails or is empty —
    common with ruled/bordered tables ('lattice' mode) in scanned reports."""
    import camelot

    tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="lattice")
    if len(tables) == 0:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="stream")
    if len(tables) == 0:
        return []
    df = tables[0].df
    df.columns = df.iloc[0]
    df = df[1:]
    return df.to_dict(orient="records")


def extract_table_json(text_as_html: str | None, pdf_path: str | None, page_number: int | None) -> list[dict]:
    if text_as_html:
        rows = html_table_to_json(text_as_html)
        if rows:
            return rows
    if pdf_path and page_number:
        return camelot_fallback(pdf_path, page_number)
    return []
