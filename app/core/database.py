from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from qdrant_client import AsyncQdrantClient
from app.core.config import settings

# SQLAlchemy setup
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Qdrant setup
qdrant_client = AsyncQdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

QDRANT_COLLECTION_NAME = "video_transcripts"

async def init_qdrant():
    collections = await qdrant_client.get_collections()
    if not any(c.name == QDRANT_COLLECTION_NAME for c in collections.collections):
        from qdrant_client.models import VectorParams, Distance
        await qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
