from datetime import datetime
from typing import get_args

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Query as SqlAlchemyQuery
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ActivityLog, AnomalySegment, Video
from backend.schemas import (
    AnomalyLabel,
    ApiResponse,
    DashboardActivityRead,
    DashboardAlertRead,
    DashboardDistributionRead,
    DashboardInvestigationRead,
    DashboardStatsRead,
    DashboardTopDetectionRead,
)


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
ANOMALY_LABELS = set(get_args(AnomalyLabel))
REVIEWABLE_VIDEO_STATUSES = ("PENDING_CONFIRM", "COMPLETED")


def _api_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "message": message},
    )


@router.get("/stats", response_model=ApiResponse[DashboardStatsRead])
def get_dashboard_stats(db: Session = Depends(get_db)) -> ApiResponse[DashboardStatsRead]:
    stats = DashboardStatsRead(
        total_videos=db.query(Video).count(),
        total_anomalies=(
            db.query(AnomalySegment)
            .filter(AnomalySegment.predicted_class != "Normal")
            .count()
        ),
        pending_reviews=(
            db.query(AnomalySegment)
            .join(Video)
            .filter(Video.status == "PENDING_CONFIRM")
            .filter(AnomalySegment.predicted_class != "Normal")
            .filter(AnomalySegment.feedback_submitted_at.is_(None))
            .count()
        ),
        reviewed_cases=(
            db.query(AnomalySegment)
            .filter(AnomalySegment.feedback_submitted_at.is_not(None))
            .count()
        ),
    )
    return ApiResponse(success=True, data=stats, message="Dashboard stats loaded")


@router.get("/distribution", response_model=ApiResponse[list[DashboardDistributionRead]])
def get_dashboard_distribution(
    db: Session = Depends(get_db),
) -> ApiResponse[list[DashboardDistributionRead]]:
    rows = (
        db.query(AnomalySegment.predicted_class, func.count(AnomalySegment.id))
        .filter(AnomalySegment.predicted_class != "Normal")
        .group_by(AnomalySegment.predicted_class)
        .order_by(func.count(AnomalySegment.id).desc(), AnomalySegment.predicted_class.asc())
        .all()
    )
    total = sum(count for _, count in rows)
    distribution = [
        DashboardDistributionRead(
            class_=predicted_class,
            count=count,
            percentage=round((count / total) * 100, 1) if total > 0 else 0,
        )
        for predicted_class, count in rows
    ]
    return ApiResponse(
        success=True,
        data=distribution,
        message="Dashboard distribution loaded",
    )


@router.get("/recent-alerts", response_model=ApiResponse[list[DashboardAlertRead]])
def get_dashboard_recent_alerts(
    anomaly_class: str | None = Query(default=None, alias="class"),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> ApiResponse[list[DashboardAlertRead]] | JSONResponse:
    selected_class = _normalize_anomaly_class(anomaly_class)
    if selected_class is False:
        return _api_error("Invalid anomaly class")

    rows = (
        _apply_segment_filters(
            db.query(AnomalySegment, Video)
            .join(Video)
            .filter(Video.status.in_(REVIEWABLE_VIDEO_STATUSES)),
            selected_class,
            date_from,
            date_to,
        )
        .order_by(AnomalySegment.created_at.desc(), AnomalySegment.id.desc())
        .limit(limit)
        .all()
    )
    alerts = [
        DashboardAlertRead(
            id=segment.id,
            video_id=video.id,
            time=_format_clock_time(segment.created_at),
            activity_type=segment.predicted_class,
            confidence=segment.confidence_score,
            anomaly_score=segment.anomaly_score,
            severity=_get_severity(segment.anomaly_score),
            review_status=segment.review_status,
            is_correct=_to_bool_or_none(segment.is_correct),
        )
        for segment, video in rows
    ]
    return ApiResponse(success=True, data=alerts, message="Dashboard recent alerts loaded")


@router.get("/top-detections", response_model=ApiResponse[list[DashboardTopDetectionRead]])
def get_dashboard_top_detections(
    anomaly_class: str | None = Query(default=None, alias="class"),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=6, ge=1, le=15),
    db: Session = Depends(get_db),
) -> ApiResponse[list[DashboardTopDetectionRead]] | JSONResponse:
    selected_class = _normalize_anomaly_class(anomaly_class)
    if selected_class is False:
        return _api_error("Invalid anomaly class")

    rows = (
        _apply_segment_filters(
            db.query(AnomalySegment)
            .join(Video)
            .with_entities(
                AnomalySegment.predicted_class,
                func.count(AnomalySegment.id),
            ),
            selected_class,
            date_from,
            date_to,
        )
        .group_by(AnomalySegment.predicted_class)
        .order_by(func.count(AnomalySegment.id).desc(), AnomalySegment.predicted_class.asc())
        .limit(limit)
        .all()
    )
    detections = [
        DashboardTopDetectionRead(class_=predicted_class, count=count)
        for predicted_class, count in rows
    ]
    return ApiResponse(
        success=True,
        data=detections,
        message="Dashboard top detections loaded",
    )


@router.get(
    "/recent-investigations",
    response_model=ApiResponse[list[DashboardInvestigationRead]],
)
def get_dashboard_recent_investigations(
    anomaly_class: str | None = Query(default=None, alias="class"),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=5, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ApiResponse[list[DashboardInvestigationRead]] | JSONResponse:
    selected_class = _normalize_anomaly_class(anomaly_class)
    if selected_class is False:
        return _api_error("Invalid anomaly class")

    video_query = _apply_video_filters(
        db.query(Video).filter(Video.status.in_(REVIEWABLE_VIDEO_STATUSES)),
        date_from,
        date_to,
    )
    if selected_class is not None:
        video_query = (
            video_query.join(AnomalySegment)
            .filter(AnomalySegment.predicted_class == selected_class)
            .distinct()
        )

    videos = (
        video_query.order_by(Video.created_at.desc(), Video.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    investigations: list[DashboardInvestigationRead] = []
    for video in videos:
        segments = sorted(video.segments, key=lambda segment: segment.segment_index)
        if selected_class is not None:
            segments = [
                segment for segment in segments if segment.predicted_class == selected_class
            ]
        if not segments:
            continue

        first_segment = segments[0]
        investigations.append(
            DashboardInvestigationRead(
                video_id=video.id,
                filename=video.filename,
                file_path=video.file_path,
                duration=float(video.duration) if video.duration is not None else None,
                file_size=video.file_size,
                detected_activity=first_segment.predicted_class,
                confidence=first_segment.confidence_score,
                anomaly_score=first_segment.anomaly_score,
                investigation_status=_get_investigation_status(video.segments),
                created_at=video.created_at,
            )
        )

    return ApiResponse(
        success=True,
        data=investigations,
        message="Dashboard recent investigations loaded",
    )


@router.get("/recent-activity", response_model=ApiResponse[list[DashboardActivityRead]])
def get_dashboard_recent_activity(
    limit: int = Query(default=5, ge=1, le=50),
    db: Session = Depends(get_db),
) -> ApiResponse[list[DashboardActivityRead]]:
    rows = (
        db.query(ActivityLog)
        .order_by(ActivityLog.created_at.desc(), ActivityLog.id.desc())
        .limit(limit)
        .all()
    )
    recent_activities = [
        DashboardActivityRead(
            type=activity.type,
            title=activity.title,
            detail=activity.description or "",
            video_id=activity.video_id,
            created_at=activity.created_at,
        )
        for activity in rows
    ]
    return ApiResponse(
        success=True,
        data=recent_activities,
        message="Dashboard recent activity loaded",
    )


def _normalize_anomaly_class(value: str | None) -> str | None | bool:
    if value is None or value == "" or value == "All":
        return None
    if value not in ANOMALY_LABELS:
        return False
    return value


def _normalize_date_start(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if len(value) == 10:
        return f"{value}T00:00:00"
    return value


def _normalize_date_end(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if len(value) == 10:
        return f"{value}T23:59:59"
    return value


def _apply_video_filters(
    query: SqlAlchemyQuery,
    date_from: str | None,
    date_to: str | None,
) -> SqlAlchemyQuery:
    start = _normalize_date_start(date_from)
    end = _normalize_date_end(date_to)
    if start is not None:
        query = query.filter(Video.created_at >= start)
    if end is not None:
        query = query.filter(Video.created_at <= end)
    return query


def _apply_segment_filters(
    query: SqlAlchemyQuery,
    anomaly_class: str | None | bool,
    date_from: str | None,
    date_to: str | None,
) -> SqlAlchemyQuery:
    query = _apply_video_filters(query, date_from, date_to)
    if anomaly_class:
        query = query.filter(AnomalySegment.predicted_class == anomaly_class)
    return query


def _get_severity(anomaly_score: float) -> str:
    if anomaly_score >= 0.85:
        return "HIGH"
    if anomaly_score >= 0.65:
        return "MEDIUM"
    return "LOW"


def _get_investigation_status(segments: list[AnomalySegment]) -> str:
    if any(segment.review_status == "PENDING_REVIEW" for segment in segments):
        return "HIGH ALERT"
    if any(segment.review_status == "CORRECTED" for segment in segments):
        return "IN REVIEW"
    return "VALIDATED"


def _to_bool_or_none(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        return datetime.min


def _format_clock_time(value: str) -> str:
    parsed = _parse_timestamp(value)
    if parsed == datetime.min:
        return value
    return parsed.strftime("%H:%M:%S")
