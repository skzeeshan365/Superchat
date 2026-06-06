import asyncio
import uuid
import os
import tempfile
import datetime
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import httpx
import re
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
import assemblyai as aai
import cohere
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.models.domain import VideoMetadata, IngestionJob, JobStatus
from app.core.database import QDRANT_COLLECTION_NAME

gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
co = cohere.AsyncClient(api_key=settings.COHERE_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def extract_video_id_from_url(url: str, platform: str) -> str:
    if platform == "youtube":
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "/shorts/" in url:
            return url.split("/shorts/")[1].split("?")[0].split("/")[0]
    elif platform == "instagram":
        if "/reel/" in url:
            return url.split("/reel/")[1].split("/")[0].split("?")[0]
        elif "/p/" in url:
            return url.split("/p/")[1].split("/")[0].split("?")[0]
    return str(uuid.uuid4())

async def extract_metadata(url: str, platform: str, video_id: str):
    if platform == "youtube" and settings.GOOGLE_CLOUD_API:
        async with httpx.AsyncClient() as client:
            api_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={settings.GOOGLE_CLOUD_API}"
            response = await client.get(api_url)
            if response.status_code == 200:
                data = response.json()
                if data.get("items"):
                    item = data["items"][0]
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    content_details = item.get("contentDetails", {})
                    
                    # Convert ISO 8601 duration
                    duration_str = content_details.get("duration", "PT0S")
                    hours, minutes, seconds = 0, 0, 0
                    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
                    if match:
                        hours = int(match.group(1) or 0)
                        minutes = int(match.group(2) or 0)
                        seconds = int(match.group(3) or 0)
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    
                    upload_date = snippet.get("publishedAt", "")[:10].replace("-", "")
                    
                    return {
                        "id": video_id,
                        "title": snippet.get("title"),
                        "uploader": snippet.get("channelTitle"),
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        "upload_date": upload_date,
                        "tags": snippet.get("tags", []),
                        "duration": total_seconds
                    }

        # Piped API fallback for metadata if Google API is missing or fails
        instances = ["https://pipedapi.kavin.rocks", "https://pipedapi.adminforge.de", "https://pipedapi.smnz.de"]
        async with httpx.AsyncClient() as client:
            for instance in instances:
                try:
                    res = await client.get(f"{instance}/streams/{video_id}", timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        upload_date = data.get("uploadDate", "")
                        if upload_date and "-" in upload_date:
                            upload_date = upload_date.replace("-", "")
                        return {
                            "id": video_id,
                            "title": data.get("title"),
                            "uploader": data.get("uploader"),
                            "view_count": data.get("views", 0),
                            "like_count": data.get("likes", 0),
                            "comment_count": 0,
                            "upload_date": upload_date,
                            "tags": [],
                            "duration": data.get("duration", 0)
                        }
                except Exception:
                    continue
        
        raise Exception("Google API is missing/invalid and all Piped proxies are down. Please add GOOGLE_CLOUD_API to Railway Variables.")

    # Instagram fallback
    def _extract():
        ydl_opts = {
            'quiet': True, 
            'skip_download': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    return await asyncio.to_thread(_extract)

async def download_audio(url: str, output_path: str, video_id: str = None):
    if video_id:
        instances = ["https://pipedapi.kavin.rocks", "https://pipedapi.adminforge.de", "https://pipedapi.smnz.de"]
        async with httpx.AsyncClient() as client:
            for instance in instances:
                try:
                    res = await client.get(f"{instance}/streams/{video_id}", timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        audio_streams = data.get("audioStreams", [])
                        if audio_streams:
                            stream_url = audio_streams[0].get("url")
                            if stream_url:
                                async with client.stream('GET', stream_url) as r:
                                    r.raise_for_status()
                                    with open(output_path, 'wb') as f:
                                        async for chunk in r.aiter_bytes(chunk_size=8192):
                                            f.write(chunk)
                                return
                except Exception:
                    continue
            
            # Cobalt API Fallback for YouTube Audio
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Origin": "https://cobalt.tools",
                "User-Agent": "Mozilla/5.0"
            }
            payload = {"url": f"https://www.youtube.com/watch?v={video_id}", "isAudioOnly": True, "aFormat": "mp3"}
            try:
                res = await client.post("https://api.cobalt.tools/api/json", json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    download_url = res.json().get("url")
                    if download_url:
                        async with client.stream('GET', download_url) as r:
                            r.raise_for_status()
                            with open(output_path, 'wb') as f:
                                async for chunk in r.aiter_bytes(chunk_size=8192):
                                    f.write(chunk)
                        return
            except Exception:
                pass
                
        raise Exception("All Proxies (Piped/Cobalt) failed to download YouTube audio. Cannot use yt-dlp due to IP blocks.")

    # Instagram Fallback
    def _download():
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    await asyncio.to_thread(_download)

async def generate_transcript_with_assemblyai(audio_path: str) -> str:
    def _generate():
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
            
        return transcript.text
    return await asyncio.to_thread(_generate)

async def get_transcript(url: str, platform: str, video_id: str) -> str:
    if platform == "youtube":
        try:
            def _get_transcript():
                return YouTubeTranscriptApi().list(video_id).find_transcript(['en']).fetch()
            transcript_list = await asyncio.to_thread(_get_transcript)
            return " ".join([t.text for t in transcript_list])
        except Exception:
            pass 
            
        # Piped API proxy fallback to bypass IP blocks
        instances = ["https://pipedapi.kavin.rocks", "https://pipedapi.adminforge.de", "https://pipedapi.smnz.de"]
        async with httpx.AsyncClient() as client:
            for instance in instances:
                try:
                    res = await client.get(f"{instance}/streams/{video_id}", timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        subtitles = data.get("subtitles", [])
                        en_sub = next((s for s in subtitles if s.get("code") == "en" and not s.get("autoGenerated")), None)
                        if not en_sub:
                            en_sub = next((s for s in subtitles if s.get("code") == "en"), None)
                        
                        if en_sub and en_sub.get("url"):
                            sub_res = await client.get(en_sub["url"], timeout=10)
                            if sub_res.status_code == 200:
                                text = sub_res.text
                                clean_text = re.sub(r'<[^>]+>', ' ', text)
                                clean_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', ' ', clean_text)
                                clean_text = " ".join(clean_text.split())
                                if clean_text.strip():
                                    return clean_text
                except Exception:
                    continue

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, f"{video_id}.m4a")
        await download_audio(url, audio_path, video_id if platform == "youtube" else None)
        transcript = await generate_transcript_with_assemblyai(audio_path)
        return transcript

async def process_video(url: str, db: AsyncSession, qdrant: AsyncQdrantClient, job_id: str):
    job = await db.get(IngestionJob, job_id)
    if not job:
        return

    try:
        job.status = JobStatus.PROCESSING
        await db.commit()

        platform = "youtube" if "youtu" in url.lower() else "instagram"
        raw_video_id = extract_video_id_from_url(url, platform)
        info = await extract_metadata(url, platform, raw_video_id)
        
        video_id = f"{platform}_{raw_video_id}"

        # Check if already processed
        existing = await db.get(VideoMetadata, video_id)
        if not existing:
            views = info.get('view_count', 0)
            likes = info.get('like_count', 0)
            comments = info.get('comment_count', 0)
            engagement_rate = ((likes + comments) / views * 100) if views > 0 else 0.0

            upload_date = info.get('upload_date')
            dt_upload = None
            if upload_date:
                dt_upload = datetime.datetime.strptime(upload_date, '%Y%m%d').replace(tzinfo=datetime.timezone.utc)

            metadata = VideoMetadata(
                id=video_id,
                platform=platform,
                url=url,
                title=info.get('title'),
                creator=info.get('uploader') or info.get('channel'),
                follower_count=info.get('channel_follower_count', 0),
                views=views,
                likes=likes,
                comments=comments,
                engagement_rate=engagement_rate,
                hashtags=info.get('tags', []),
                upload_date=dt_upload,
                duration=info.get('duration', 0)
            )
            db.add(metadata)
            
            transcript = await get_transcript(url, platform, raw_video_id)
            
            # Chunk and embed
            chunks = text_splitter.split_text(transcript)
            if chunks:
                txt_resp = await co.embed(
                    model="embed-v4.0",
                    texts=chunks,
                    input_type="search_document",
                    embedding_types=["float"]
                )
                embedded = txt_resp.embeddings.float
                
                points = []
                for i, (chunk, vector) in enumerate(zip(chunks, embedded)):
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "video_id": video_id,
                            "platform": platform,
                            "text": chunk,
                            "chunk_index": i
                        }
                    ))
                
                await qdrant.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=points
                )
        
        job.status = JobStatus.COMPLETED
        await db.commit()

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        await db.commit()
