import cv2
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import DATABASE_PATH, get_db
from backend.models import ActivityLog, AnomalySegment, Batch, ProcessingJob, User, Video
from backend.schemas import (
    AnomalySegmentRead,
    ApiResponse,
    BatchRead,
    UploadBatchRead,
    VideoDetailRead,
    VideoDurationProbeRead,
    VideoRead,
)
from backend.utils import format_time, generate_batch_id, generate_video_id, vietnam_now_iso


router = APIRouter(prefix="/api/videos", tags=["videos"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_FILES_PER_BATCH = 3
MAX_BATCH_SIZE_BYTES = 300 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 300.0


class VideoDurationError(ValueError):
    pass


class VideoDurationLimitError(ValueError):
    pass


def _api_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "message": message},
    )


def _get_upload_size(file: UploadFile) -> int:
    current_position = file.file.tell()
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(current_position)
    return size


def _safe_unlink(path: Path) -> None:
    for attempt in range(3):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 2:
                return
            sleep(0.1)
        except OSError:
            return


def _read_video_duration(video_path: Path) -> float:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoDurationError("Could not open video file to read duration")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if fps <= 0 or frame_count <= 0:
            raise VideoDurationError("Video metadata (fps/frame count) is missing or invalid")
            
        duration = float(frame_count) / float(fps)
    except VideoDurationError:
        raise
    except Exception as exc:
        raise VideoDurationError(f"Could not read video duration: {exc}")
    finally:
        if 'cap' in locals() and cap is not None and cap.isOpened():
            cap.release()

    if not math.isfinite(duration) or duration <= 0:
        raise VideoDurationError("Video duration metadata is missing or invalid")

    return duration


def _form_value(values: list[str] | None, index: int) -> str | None:
    if values is None or index >= len(values):
        return None

    value = values[index].strip()
    return value or None


def _create_unique_batch_id(db: Session) -> str:
    batch_id = generate_batch_id()
    while db.get(Batch, batch_id) is not None:
        sleep(1)
        batch_id = generate_batch_id()
    return batch_id


def _create_unique_video_id(db: Session) -> str:
    video_id = generate_video_id()
    while db.get(Video, video_id) is not None:
        video_id = generate_video_id()
    return video_id


def _start_worker_loop_if_available(db_path: str) -> None:
    try:
        from backend.worker import start_worker_loop
    except ModuleNotFoundError as exc:
        if exc.name == "backend.worker":
            return
        raise

    start_worker_loop(db_path)


@router.get("", response_model=ApiResponse[list[VideoRead]])
def list_videos(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[list[VideoRead]]:
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    return ApiResponse(
        success=True,
        data=[VideoRead.model_validate(video) for video in videos],
        message="Videos loaded",
    )


@router.post("/upload", response_model=ApiResponse[UploadBatchRead])
def upload_videos(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()],
    names: Annotated[list[str] | None, Form()] = None,
    descriptions: Annotated[list[str] | None, Form()] = None,
    locations: Annotated[list[str] | None, Form()] = None,
    durations: Annotated[list[float] | None, Form()] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[UploadBatchRead] | JSONResponse:
    if not files:
        return _api_error("At least one video file is required")

    if len(files) > MAX_FILES_PER_BATCH:
        return _api_error("A batch can contain at most 3 videos")

    file_sizes: list[int] = []
    extensions: list[str] = []
    for file in files:
        if not file.filename:
            return _api_error("Every uploaded file must have a filename")

        extension = Path(file.filename).suffix.lower()
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            return _api_error(f"Unsupported video format: {file.filename}")

        file_size = _get_upload_size(file)
        file_sizes.append(file_size)
        extensions.append(extension)

    if sum(file_sizes) > MAX_BATCH_SIZE_BYTES:
        return _api_error("Total batch size exceeds 300 MB")

    UPLOAD_DIR.mkdir(exist_ok=True)
    now = vietnam_now_iso()
    batch_id = _create_unique_batch_id(db)
    first_filename = files[0].filename or "batch"
    batch_name = Path(first_filename).stem
    saved_paths: list[Path] = []
    videos: list[Video] = []

    try:
        batch = Batch(
            id=batch_id,
            name=batch_name,
            total_videos=len(files),
            created_at=now,
        )
        db.add(batch)

        for index, file in enumerate(files):
            video_id = _create_unique_video_id(db)
            extension = extensions[index]
            stored_filename = f"{video_id}{extension}"
            stored_path = UPLOAD_DIR / stored_filename
            relative_file_path = f"uploads/{stored_filename}"

            file.file.seek(0)
            with stored_path.open("wb") as output:
                output.write(file.file.read())
            saved_paths.append(stored_path)

            duration = _read_video_duration(stored_path)
            if duration > MAX_VIDEO_DURATION_SECONDS:
                raise VideoDurationLimitError(f"Video exceeds 5 minute limit: {file.filename}")

            original_filename = file.filename or stored_filename
            video = Video(
                id=video_id,
                batch_id=batch_id,
                filename=original_filename,
                name=_form_value(names, index) or Path(original_filename).stem,
                description=_form_value(descriptions, index),
                location=_form_value(locations, index),
                file_path=relative_file_path,
                file_size=file_sizes[index],
                duration=duration,
                status="WAITING",
                progress_step="WAITING",
                created_at=now,
                updated_at=now,
            )
            videos.append(video)
            db.add(video)

            db.add(
                ProcessingJob(
                    video_id=video_id,
                    status="PENDING",
                    created_at=now,
                )
            )
            db.add(
                ActivityLog(
                    type="UPLOAD",
                    title="Video uploaded",
                    description=video.name,
                    video_id=video_id,
                    created_at=now,
                )
            )

        db.commit()
    except VideoDurationLimitError as exc:
        db.rollback()
        for path in saved_paths:
            if path.exists():
                _safe_unlink(path)
        return _api_error(str(exc), status_code=400)
    except VideoDurationError as exc:
        db.rollback()
        for path in saved_paths:
            if path.exists():
                _safe_unlink(path)
        return _api_error(f"Could not read video duration: {exc}", status_code=400)
    except (OSError, SQLAlchemyError) as exc:
        db.rollback()
        for path in saved_paths:
            if path.exists():
                _safe_unlink(path)
        return _api_error(f"Upload failed: {exc}", status_code=500)

    db.refresh(batch)
    for video in videos:
        db.refresh(video)

    background_tasks.add_task(_start_worker_loop_if_available, str(DATABASE_PATH))

    return ApiResponse(
        success=True,
        data=UploadBatchRead(
            batch=BatchRead.model_validate(batch),
            videos=[VideoRead.model_validate(video) for video in videos],
        ),
        message="Upload successful",
    )


@router.post("/probe-duration", response_model=ApiResponse[VideoDurationProbeRead])
def probe_video_duration(
    file: Annotated[UploadFile, File()],
    current_user: User = Depends(get_current_user),
) -> ApiResponse[VideoDurationProbeRead] | JSONResponse:
    if not file.filename:
        return _api_error("Video file must have a filename")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return _api_error(f"Unsupported video format: {file.filename}")

    if _get_upload_size(file) > MAX_BATCH_SIZE_BYTES:
        return _api_error("Total batch size exceeds 300 MB")

    UPLOAD_DIR.mkdir(exist_ok=True)
    temp_path: Path | None = None

    try:
        with NamedTemporaryFile(dir=UPLOAD_DIR, suffix=extension, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            file.file.seek(0)
            temp_file.write(file.file.read())

        duration = _read_video_duration(temp_path)
    except VideoDurationError as exc:
        return _api_error(f"Could not read video duration: {exc}", status_code=400)
    except OSError as exc:
        return _api_error(f"Could not read video duration: {exc}", status_code=500)
    finally:
        if temp_path is not None and temp_path.exists():
            _safe_unlink(temp_path)

    return ApiResponse(
        success=True,
        data=VideoDurationProbeRead(filename=file.filename, duration=duration),
        message="Video duration loaded",
    )


@router.post("/{video_id}/retry", response_model=ApiResponse[VideoRead])
def retry_video(
    video_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[VideoRead] | JSONResponse:
    video = db.get(Video, video_id)
    if video is None:
        return _api_error("Video not found", status_code=404)

    if video.status != "FAILED":
        return _api_error("Only FAILED videos can be retried")

    now = vietnam_now_iso()
    video.status = "WAITING"
    video.progress_step = "WAITING"
    video.error_message = None
    video.updated_at = now

    job = db.query(ProcessingJob).filter(ProcessingJob.video_id == video_id).first()
    if job is None:
        job = ProcessingJob(video_id=video_id, status="PENDING", created_at=now)
        db.add(job)
    else:
        job.status = "PENDING"
        job.started_at = None
        job.finished_at = None

    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        return _api_error(f"Retry failed: {exc}", status_code=500)

    db.refresh(video)
    background_tasks.add_task(_start_worker_loop_if_available, str(DATABASE_PATH))
    return ApiResponse(
        success=True,
        data=VideoRead.model_validate(video),
        message="Retry queued",
    )


@router.get("/{video_id}/export")
def export_video_report(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    video = db.get(Video, video_id)
    if video is None:
        return _api_error("Video not found", status_code=404)

    segments = (
        db.query(AnomalySegment)
        .filter(AnomalySegment.video_id == video_id)
        .order_by(AnomalySegment.segment_index.asc())
        .all()
    )
    feedback_submitted = sum(1 for segment in segments if segment.feedback_submitted_at is not None)
    pending_review = sum(1 for segment in segments if segment.review_status == "PENDING_REVIEW")
    report = {
        "video": {
            "id": video.id,
            "name": video.name,
            "location": video.location,
            "duration": video.duration,
            "status": video.status,
            "created_at": video.created_at,
        },
        "summary": {
            "total_segments": len(segments),
            "total_anomalies": sum(1 for segment in segments if segment.predicted_class != "Normal"),
            "feedback_submitted": feedback_submitted,
            "pending_review": pending_review,
        },
        "segments": [_build_report_segment(segment) for segment in segments],
    }
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f'attachment; filename="report_{video_id}.json"'},
    )


@router.get("/{video_id}", response_model=ApiResponse[VideoDetailRead])
def get_video_detail(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[VideoDetailRead] | JSONResponse:
    video = db.get(Video, video_id)
    if video is None:
        return _api_error("Video not found", status_code=404)

    segments = (
        db.query(AnomalySegment)
        .filter(AnomalySegment.video_id == video_id)
        .order_by(AnomalySegment.segment_index.asc())
        .all()
    )
    video_data = VideoRead.model_validate(video).model_dump()
    return ApiResponse(
        success=True,
        data=VideoDetailRead(
            **video_data,
            segments=[AnomalySegmentRead.model_validate(segment) for segment in segments],
        ),
        message="Video loaded",
    )


def _build_report_segment(segment: AnomalySegment) -> dict:
    return {
        "segment_id": f"SEG-{segment.segment_index + 1:04d}",
        "time_range": f"{format_time(segment.start_time)} - {format_time(segment.end_time)}",
        "predicted_class": segment.predicted_class,
        "confidence_score": segment.confidence_score,
        "anomaly_score": segment.anomaly_score,
        "review_status": segment.review_status,
        "is_correct": None if segment.is_correct is None else bool(segment.is_correct),
        "verified_label": segment.verified_label,
        "other_description": segment.other_description,
        "investigator_comment": segment.investigator_comment,
        "feedback_submitted_at": segment.feedback_submitted_at,
    }
