from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    GEMINI_API_KEY: str
    COHERE_API_KEY: str
    ASSEMBLYAI_API_KEY: str
    POSTGRES_CHECKPOINT_URL: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
