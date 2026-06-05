import enum
from sqlalchemy import Column, String, Integer, Float, DateTime, Enum, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class VideoMetadata(Base):
    __tablename__ = "video_metadata"

    id = Column(String, primary_key=True, index=True) # video ID
    platform = Column(String, index=True) # 'youtube' or 'instagram'
    url = Column(String)
    title = Column(String, nullable=True)
    creator = Column(String, nullable=True)
    follower_count = Column(Integer, nullable=True)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)
    hashtags = Column(JSON, nullable=True)
    upload_date = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Integer, nullable=True) # in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
