from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    filters: dict | None = None
    use_reranker: bool = True


class IngestRequest(BaseModel):
    file_path: str
    department: str
    classification: str = "internal"
    access_tags: list[str] = []
    extra_metadata: dict = {}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
