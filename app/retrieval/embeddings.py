from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts. OpenAI's embedding endpoint handles batches of
    up to ~2048 inputs; for 100k+ doc corpora, callers should chunk their
    own batches (e.g. 100 at a time) to stay well under token/size limits."""
    if not texts:
        return []
    response = _client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
