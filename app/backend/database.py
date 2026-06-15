from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy import event, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "anomaly.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _configure_sqlite_connection(dbapi_connection, connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=TRUNCATE")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db() -> None:
    from backend import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _backfill_activity_log()


def _backfill_activity_log() -> None:
    from backend.models import ActivityLog, AnomalySegment, Video

    db = SessionLocal()
    try:
        videos = db.query(Video).all()
        for video in videos:
            if not _activity_exists(db, video.id, "UPLOAD"):
                db.add(
                    ActivityLog(
                        type="UPLOAD",
                        title="Video uploaded",
                        description=video.name,
                        video_id=video.id,
                        created_at=video.created_at,
                    )
                )

            first_feedback_at = (
                db.query(func.min(AnomalySegment.feedback_submitted_at))
                .filter(
                    AnomalySegment.video_id == video.id,
                    AnomalySegment.feedback_submitted_at.is_not(None),
                )
                .scalar()
            )
            if first_feedback_at is not None and not _activity_exists(
                db, video.id, "REVIEW_COMPLETE"
            ):
                db.add(
                    ActivityLog(
                        type="REVIEW_COMPLETE",
                        title="Feedback submitted",
                        description=video.name,
                        video_id=video.id,
                        created_at=first_feedback_at,
                    )
                )

            flagged_count = (
                db.query(AnomalySegment)
                .filter(
                    AnomalySegment.video_id == video.id,
                    AnomalySegment.anomaly_score > 0.8,
                )
                .count()
            )
            if flagged_count > 0 and not _activity_exists(db, video.id, "FLAG"):
                db.add(
                    ActivityLog(
                        type="FLAG",
                        title="High anomaly flagged",
                        description=(
                            f"{flagged_count} high-risk segment(s) detected in {video.name}"
                        ),
                        video_id=video.id,
                        created_at=video.updated_at,
                    )
                )

        db.commit()
    finally:
        db.close()


def _activity_exists(db: Session, video_id: str, activity_type: str) -> bool:
    from backend.models import ActivityLog

    return (
        db.query(ActivityLog)
        .filter(ActivityLog.video_id == video_id, ActivityLog.type == activity_type)
        .first()
        is not None
    )
