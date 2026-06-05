from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router as api_router
from app.core.database import engine, Base, init_qdrant

app = FastAPI(title="Social Media RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Initialize DB schemas
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Initialize Qdrant collection
    await init_qdrant()
    # Open connection pool for LangGraph PostgresSaver
    from app.api.v1.endpoints import pool
    await pool.open()

app.include_router(api_router, prefix="/api/v1")
