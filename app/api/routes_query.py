import json
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import QueryRequest
from app.db.session import get_db
from app.db.models import User
from app.auth.jwt_handler import get_current_user
from app.auth.rbac import build_access_context
from app.generation.rag_engine import answer_query_stream

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/stream")
async def query_stream(
    req: QueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Server-Sent Events endpoint. Each event is a JSON payload:
    {"type": "chunk", "text": "..."} while the answer streams,
    {"type": "sources", "sources": [...]} once generation finishes,
    {"type": "done", "latency_ms": N} to close out.
    """
    access_ctx = await build_access_context(db, user)

    async def event_generator():
        async for event in answer_query_stream(req.query, access_ctx, req.filters, req.use_reranker):
            yield {"event": event["type"], "data": json.dumps(event)}

    return EventSourceResponse(event_generator())
