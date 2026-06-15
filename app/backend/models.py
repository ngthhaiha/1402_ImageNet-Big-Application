from sqlalchemy import Column, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import REAL

from backend.database import Base


class Batch(Base):
    __tablename__ = "batches"

    id = Column(Text, primary_key=True)
    name = Column(Text)
    total_videos = Column(Integer, nullable=False)
    created_at = Column(Text, nullable=False)

    videos = relationship("Video", back_populates="batch")


class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        Index("idx_videos_status", "status"),
        Index("idx_videos_batch_id", "batch_id"),
        Index("idx_videos_created_at", "created_at"),
    )

    id = Column(Text, primary_key=True)
    batch_id = Column(Text, ForeignKey("batches.id"))
    filename = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text)
    location = Column(Text)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer)
    duration = Column(REAL)
    status = Column(Text, nullable=False, default="WAITING", server_default="WAITING")
    progress_step = Column(Text, nullable=False, default="WAITING", server_default="WAITING")
    error_message = Column(Text)
    created_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False)

    batch = relationship("Batch", back_populates="videos")
    processing_job = relationship("ProcessingJob", back_populates="video", uselist=False)
    segments = relationship("AnomalySegment", back_populates="video")
    activity_logs = relationship("ActivityLog", back_populates="video")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index("idx_jobs_video_id", "video_id"),
        Index("idx_jobs_status", "status"),
        {"sqlite_autoincrement": True},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Text, ForeignKey("videos.id"), nullable=False, unique=True)
    status = Column(Text, nullable=False, default="PENDING", server_default="PENDING")
    started_at = Column(Text)
    finished_at = Column(Text)
    created_at = Column(Text, nullable=False)

    video = relationship("Video", back_populates="processing_job")


class AnomalySegment(Base):
    __tablename__ = "anomaly_segments"
    __table_args__ = (
        Index("idx_segments_video_id", "video_id"),
        {"sqlite_autoincrement": True},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Text, ForeignKey("videos.id"), nullable=False)
    segment_index = Column(Integer, nullable=False)
    start_time = Column(REAL, nullable=False)
    end_time = Column(REAL, nullable=False)
    anomaly_score = Column(REAL, nullable=False)
    predicted_class = Column(Text, nullable=False)
    confidence_score = Column(REAL, nullable=False)
    is_correct = Column(Integer)
    verified_label = Column(Text)
    other_description = Column(Text)
    investigator_comment = Column(Text)
    feedback_submitted_at = Column(Text)
    review_status = Column(
        Text,
        nullable=False,
        default="PENDING_REVIEW",
        server_default="PENDING_REVIEW",
    )
    created_at = Column(Text, nullable=False)

    video = relationship("Video", back_populates="segments")


class ActivityLog(Base):
    __tablename__ = "activity_log"
    __table_args__ = (
        Index("idx_activity_log_created_at", "created_at"),
        Index("idx_activity_log_video_id", "video_id"),
        {"sqlite_autoincrement": True},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text)
    video_id = Column(Text, ForeignKey("videos.id"))
    created_at = Column(Text, nullable=False)

    video = relationship("Video", back_populates="activity_logs")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    target_url = Column(String, nullable=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=True)
    is_read = Column(Integer, nullable=False, default=0)
    created_at = Column(String, nullable=False)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("idx_users_username", "username", unique=True),
        Index("idx_users_email", "email", unique=True),
        {"sqlite_autoincrement": True},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(Text, nullable=False, unique=True)
    email = Column(Text, nullable=False, unique=True)
    password_hash = Column(Text, nullable=False)
    created_at = Column(Text, nullable=False)
