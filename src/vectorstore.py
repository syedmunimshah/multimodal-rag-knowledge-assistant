from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from src.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vectorstore() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.pg_collection_name,
        connection=settings.pg_connection_string,
        use_jsonb=True,
    )
