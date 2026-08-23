import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pg_connection_string: str = os.getenv("PG_CONNECTION_STRING", "")
    pg_collection_name: str = os.getenv("PG_COLLECTION_NAME", "knowledge_base")

    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o-mini")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_reasoning_hops: int = int(os.getenv("MAX_REASONING_HOPS", "3"))


settings = Settings()
