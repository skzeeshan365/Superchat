# Superchat RAG Backend

This is the backend for the Superchat social media RAG chatbot assignment. It ingests YouTube and Instagram videos, computes engagement metrics, transcribes audio (using Gemini Flash 3), creates chunks, and embeds them using Cohere v4. The chat is powered by LangGraph to retrieve stats and transcripts.

## Setup

1. **Environment**: Copy `.env.example` to `.env` and fill in your Gemini and Cohere API keys.
2. **Infrastructure**: Start PostgreSQL and Qdrant locally:
   ```bash
   docker-compose up -d
   ```
3. **Dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Run Server**:
   ```bash
   uvicorn app.main:app --reload
   ```

## Scale & Trade-offs (Assignment Notes)

This architecture easily supports ~1,000 creators/day using FastAPI's lightweight `BackgroundTasks` instead of heavy message queues. However, for 10,000 users, several bottlenecks emerge:
1. `BackgroundTasks` will exhaust container memory. Migration to Celery/ARQ is needed.
2. Instagram scraping via `yt-dlp` from a single IP will be rate-limited/banned.
3. Database connections will be exhausted, requiring PgBouncer.
