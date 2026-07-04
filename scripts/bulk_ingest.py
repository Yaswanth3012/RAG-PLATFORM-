"""
Bulk ingestion for 100,000+ document corpora.

Usage:
    python scripts/bulk_ingest.py --root /data/corpus --department finance --classification confidential

Runs ingestion concurrently with a bounded worker pool (I/O + API-bound:
embedding calls and vision captioning dominate cost, not CPU), and logs
failures to a manifest file so a re-run only retries what failed —
essential at this scale since a single bad PDF shouldn't kill a nightly batch job.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from app.db.session import AsyncSessionLocal
from app.ingestion.pipeline import ingest_document

CONCURRENCY = 8
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".png", ".jpg", ".jpeg"}


async def ingest_one(sem: asyncio.Semaphore, file_path: str, department: str, classification: str, tags: list[str], failures: list):
    async with sem:
        async with AsyncSessionLocal() as db:
            try:
                await ingest_document(
                    db=db, file_path=file_path, department=department,
                    classification=classification, access_tags=tags, ingested_by="system",
                )
                print(f"OK   {file_path}")
            except Exception as e:
                print(f"FAIL {file_path}: {e}")
                failures.append({"file_path": file_path, "error": str(e)})


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--department", required=True)
    parser.add_argument("--classification", default="internal")
    parser.add_argument("--tags", default="")
    parser.add_argument("--manifest-out", default="ingest_failures.json")
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or [
        f"dept:{args.department}", f"classification:{args.classification}"
    ]

    files = [
        str(p) for p in Path(args.root).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"Found {len(files)} files to ingest under {args.root}")

    sem = asyncio.Semaphore(CONCURRENCY)
    failures: list = []
    await asyncio.gather(*[
        ingest_one(sem, f, args.department, args.classification, tags, failures) for f in files
    ])

    if failures:
        with open(args.manifest_out, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"{len(failures)} failures written to {args.manifest_out}")


if __name__ == "__main__":
    asyncio.run(main())
