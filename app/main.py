import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.retrieval.qdrant_store import ensure_collection
from app.api import routes_query, routes_ingest, routes_admin

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield


app = FastAPI(title="Enterprise RAG Platform", version="1.0.0", lifespan=lifespan)

app.include_router(routes_admin.router)
app.include_router(routes_query.router)
app.include_router(routes_ingest.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
