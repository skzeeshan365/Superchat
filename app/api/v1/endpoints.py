import uuid
import json
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage

from app.core.database import get_db, qdrant_client
from app.models.domain import IngestionJob, JobStatus, VideoMetadata
from app.services.ingestion import process_video
from app.graph.agent import graph
from app.core.config import settings

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

router = APIRouter()

class IngestRequest(BaseModel):
    urls: List[str]

class ChatRequest(BaseModel):
    session_id: str
    video_ids: List[str]
    message: str

@router.post("/ingest")
async def ingest_videos(request: IngestRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    if len(request.urls) != 2:
        raise HTTPException(status_code=400, detail="Exactly two URLs are required.")

    job_ids = []
    for url in request.urls:
        job_id = str(uuid.uuid4())
        job = IngestionJob(id=job_id, status=JobStatus.PENDING)
        db.add(job)
        job_ids.append(job_id)
        
        background_tasks.add_task(process_video, url, db, qdrant_client, job_id)
        
    await db.commit()
    return {"job_ids": job_ids}

@router.get("/ingest/{job_id}")
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status.value, "error": job.error_message}


# Pool for the LangGraph checkpointer
pool = AsyncConnectionPool(
    conninfo=settings.POSTGRES_CHECKPOINT_URL,
    max_size=20,
    kwargs={"autocommit": True},
    open=False
)

from app.services.ingestion import extract_video_id_from_url

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_stream():
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        
        parsed_video_ids = []
        for vid in request.video_ids:
            # If the frontend sent a URL, we need to convert it to the ID format used in the DB
            if "http" in vid:
                platform = "youtube" if "youtu" in vid.lower() else "instagram"
                raw_id = extract_video_id_from_url(vid, platform)
                parsed_video_ids.append(f"{platform}_{raw_id}")
            else:
                parsed_video_ids.append(vid)

        config = {"configurable": {"thread_id": request.session_id}}
        state = {
            "messages": [HumanMessage(content=request.message)],
            "video_ids": parsed_video_ids
        }
        
        try:
            async for event in graph.astream_events(state, config, version="v2", checkpointer=checkpointer):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        if isinstance(content, list):
                            text_parts = []
                            for block in content:
                                if isinstance(block, str):
                                    text_parts.append(block)
                                elif isinstance(block, dict) and "text" in block:
                                    text_parts.append(block["text"])
                            content = "".join(text_parts)
                            
                        if isinstance(content, str) and content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'event': 'tool_start', 'name': event['name']})}\n\n"
                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'event': 'tool_end', 'name': event['name']})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
