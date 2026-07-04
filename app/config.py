from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    llm_model: str = "claude-sonnet-4-6"

    database_url: str = "postgresql+asyncpg://rag:rag_password@localhost:5432/rag_platform"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_docs"

    retrieval_backend: str = "bm25"  # bm25 | elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    dense_top_k: int = 40
    sparse_top_k: int = 40
    fused_top_k: int = 8
    rrf_k: int = 60

    jwt_secret: str = "change_me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
