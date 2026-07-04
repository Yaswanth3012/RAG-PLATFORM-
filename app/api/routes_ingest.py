from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import shutil, uuid, os

from app.db.session import get_db
from app.db.models import User
from app.auth.jwt_handler import get_current_user
from app.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("")
async def ingest(
    file: UploadFile = File(...),
    department: str = Form(...),
    classification: str = Form("internal"),
    access_tags: str = Form(""),  # comma-separated, e.g. "dept:finance,classification:confidential"
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    role_names = {r.name for r in user.roles}
    if "admin" not in role_names and "ingestor" not in role_names:
        raise HTTPException(status_code=403, detail="Not permitted to ingest documents")

    dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    tags = [t.strip() for t in access_tags.split(",") if t.strip()] or [
        f"dept:{department}", f"classification:{classification}"
    ]

    doc = await ingest_document(
        db=db,
        file_path=dest_path,
        department=department,
        classification=classification,
        access_tags=tags,
        ingested_by=str(user.id),
    )
    return {"document_id": str(doc.id), "status": doc.status, "title": doc.title}
