import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import select
from app.core.config import settings
from app.core.database import AsyncSessionLocal, QDRANT_COLLECTION_NAME
from app.models.domain import VideoMetadata
from langchain_cohere import CohereEmbeddings
import asyncio

embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=settings.COHERE_API_KEY)

# Use synchronous Qdrant client for LangChain tools (or async tools, let's use async)
from qdrant_client import AsyncQdrantClient
qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    video_ids: List[str]

@tool
async def get_video_stats(video_ids: list[str]) -> str:
    """Useful to get exact stats like views, likes, comments, engagement rate, creator, and follower count for specific videos."""
    async with AsyncSessionLocal() as session:
        stmt = select(VideoMetadata).where(VideoMetadata.id.in_(video_ids))
        result = await session.execute(stmt)
        videos = result.scalars().all()
        
        if not videos:
            return "No statistics found for the provided video IDs."
            
        stats = []
        for v in videos:
            stats.append(
                f"Video ID: {v.id}\n"
                f"Creator: {v.creator} (Followers: {v.follower_count})\n"
                f"Views: {v.views}, Likes: {v.likes}, Comments: {v.comments}\n"
                f"Engagement Rate: {v.engagement_rate:.2f}%\n"
                f"Duration: {v.duration}s\n"
            )
        return "\n".join(stats)

@tool
async def search_video_transcripts(query: str, video_ids: list[str]) -> str:
    """Useful to search through the spoken words/transcripts of the videos to compare hooks, topics, or what worked."""
    # Embed the query
    vector = await embeddings.aembed_query(query)
    
    # Filter by video_ids
    should_conditions = [
        FieldCondition(key="video_id", match=MatchValue(value=vid)) for vid in video_ids
    ]
    
    response = await qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=vector,
        query_filter=Filter(should=should_conditions) if should_conditions else None,
        limit=5
    )
    results = response.points
    
    if not results:
        return "No relevant transcript sections found."
        
    contexts = []
    for r in results:
        payload = r.payload or {}
        vid = payload.get('video_id', 'Unknown')
        text = payload.get('text', '')
        # Simple citation
        contexts.append(f"[Source: {vid}] {text}")
        
    return "\n\n".join(contexts)

tools = [get_video_stats, search_video_transcripts]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a social media analysis assistant.
You help creators understand why certain videos perform better by analyzing their stats and transcripts.
Always cite your sources using the format [Source: VideoID] when referencing transcript quotes.
Be concise and analytical."""

async def agent_node(state: AgentState):
    messages = state["messages"]
    video_ids = state.get("video_ids", [])
    
    # Strip any existing system messages to avoid duplicates
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Inject the video_ids context into the system prompt
    dynamic_prompt = system_prompt + f"\n\nThe user is currently inquiring about the following Video IDs: {', '.join(video_ids)}.\nYou MUST use your tools to fetch stats or search transcripts using these exact IDs. Do NOT ask the user for the video IDs."
    
    invoke_messages = [SystemMessage(content=dynamic_prompt)] + filtered_messages

    response = await llm_with_tools.ainvoke(invoke_messages)
    return {"messages": [response]}

# Define Graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
# Add conditional edge for tool calling
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")

# Compile
graph = builder.compile()
