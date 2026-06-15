from datetime import datetime, timedelta
from typing import get_args

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import AnomalySegment, User, Video
from backend.segment_groups import SegmentGroup, group_segment_rows
from backend.schemas import (
    AlertLogItem,
    AlertLogResponse,
    AlertStats,
    AnomalyLabel,
    ApiResponse,
    CriticalAlertItem,
    DistributionItem,
    Severity,
)
from backend.utils import VIETNAM_TIMEZONE, vietnam_now


router = APIRouter(prefix="/api/alerts", tags=["alerts"])
ANOMALY_LABELS = set(get_args(AnomalyLabel))
SEVERITIES = set(get_args(Severity))


def _api_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "message": message},
    )


@router.get("/stats", response_model=ApiResponse[AlertStats])
def get_alert_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[AlertStats]:
    stats = AlertStats(
        total_alerts=db.query(func.count(AnomalySegment.id)).scalar() or 0,
        high_severity=(
            db.query(func.count(AnomalySegment.id))
            .filter(AnomalySegment.anomaly_score >= 0.85)
            .scalar()
            or 0
        ),
        pending_reviews=(
            db.query(func.count(AnomalySegment.id))
            .filter(AnomalySegment.feedback_submitted_at.is_(None))
            .scalar()
            or 0
        ),
        reviewed_alerts=(
            db.query(func.count(AnomalySegment.id))
            .filter(AnomalySegment.feedback_submitted_at.is_not(None))
            .scalar()
            or 0
        ),
    )
    return ApiResponse(success=True, data=stats, message="Alert stats loaded")


@router.get("/log", response_model=ApiResponse[AlertLogResponse])
def get_alert_log(
    name: str | None = Query(default=None),
    activity: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    status: str | None = Query(default=None),
    alert_date: str | None = Query(default=None, alias="date"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[AlertLogResponse] | JSONResponse:
    selected_activity = _normalize_activity(activity)
    if selected_activity is False:
        return _api_error("Invalid activity")

    selected_severity = _normalize_severity(severity)
    if selected_severity is False:
        return _api_error("Invalid severity")

    selected_status = _normalize_status(status)
    if selected_status is False:
        return _api_error("Invalid status")

    query = db.query(AnomalySegment, Video).join(Video)
    query = _apply_log_filters(
        query=query,
        name=name,
        activity=selected_activity,
        severity=selected_severity,
        status=selected_status,
        alert_date=alert_date,
    )

    rows = (
        query.order_by(AnomalySegment.created_at.desc(), AnomalySegment.id.desc())
        .all()
    )
    groups = _sort_groups(group_segment_rows(rows))
    total = len(groups)
    paginated_groups = groups[(page - 1) * limit : page * limit]
    response = AlertLogResponse(
        items=[_to_alert_log_item(group) for group in paginated_groups],
        total=total,
        page=page,
        total_pages=((total + limit - 1) // limit) if total > 0 else 0,
    )
    return ApiResponse(success=True, data=response, message="Alert log loaded")


@router.get("/distribution", response_model=ApiResponse[list[DistributionItem]])
def get_alert_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[list[DistributionItem]]:
    rows = (
        db.query(AnomalySegment.predicted_class, func.count(AnomalySegment.id))
        .group_by(AnomalySegment.predicted_class)
        .order_by(func.count(AnomalySegment.id).desc(), AnomalySegment.predicted_class.asc())
        .all()
    )
    total = sum(count for _, count in rows)
    distribution = [
        DistributionItem(
            predicted_class=predicted_class,
            count=count,
            percentage=round((count / total) * 100, 1) if total > 0 else 0,
        )
        for predicted_class, count in rows[:5]
    ]
    return ApiResponse(
        success=True,
        data=distribution,
        message="Alert distribution loaded",
    )


@router.get("/critical", response_model=ApiResponse[list[CriticalAlertItem]])
def get_critical_alerts(
    limit: int = Query(default=10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[list[CriticalAlertItem]]:
    rows = (
        db.query(AnomalySegment, Video)
        .join(Video)
        .filter(AnomalySegment.anomaly_score >= 0.85)
        .order_by(AnomalySegment.created_at.desc(), AnomalySegment.id.desc())
        .all()
    )
    alerts = [_to_critical_alert_item(group) for group in _sort_groups(group_segment_rows(rows))[:limit]]
    return ApiResponse(success=True, data=alerts, message="Critical alerts loaded")


def _apply_log_filters(
    query,
    name: str | None,
    activity: str | None | bool,
    severity: str | None | bool,
    status: str | None | bool,
    alert_date: str | None,
):
    if name is not None and name.strip() != "":
        query = query.filter(Video.filename.like(f"%{name.strip()}%"))
    if activity:
        query = query.filter(AnomalySegment.predicted_class == activity)
    if severity:
        query = _apply_severity_filter(query, severity)
    if status == "PENDING_REVIEW":
        query = query.filter(AnomalySegment.feedback_submitted_at.is_(None))
    elif status == "REVIEWED":
        query = query.filter(AnomalySegment.feedback_submitted_at.is_not(None))
    if alert_date is not None and alert_date.strip() != "":
        query = query.filter(func.date(AnomalySegment.created_at) == alert_date.strip())
    return query


def _apply_severity_filter(query, severity: str):
    if severity == "HIGH":
        return query.filter(AnomalySegment.anomaly_score >= 0.85)
    if severity == "MEDIUM":
        return query.filter(
            AnomalySegment.anomaly_score >= 0.65,
            AnomalySegment.anomaly_score < 0.85,
        )
    return query.filter(AnomalySegment.anomaly_score < 0.65)


def _normalize_activity(value: str | None) -> str | None | bool:
    if value is None or value == "" or value == "All":
        return None
    if value not in ANOMALY_LABELS:
        return False
    return value


def _normalize_severity(value: str | None) -> str | None | bool:
    if value is None or value == "" or value == "All":
        return None
    normalized = value.upper()
    if normalized not in SEVERITIES:
        return False
    return normalized


def _normalize_status(value: str | None) -> str | None | bool:
    if value is None or value == "" or value == "All":
        return None
    normalized = value.upper()
    if normalized in {"PENDING_REVIEW", "UNREVIEWED", "PENDING"}:
        return "PENDING_REVIEW"
    if normalized in {"REVIEWED", "LABEL_CORRECT", "CORRECTED", "LOGGED"}:
        return "REVIEWED"
    return False


def _sort_groups(groups: list[SegmentGroup]) -> list[SegmentGroup]:
    return sorted(
        groups,
        key=lambda group: (group.created_at, group.sort_id),
        reverse=True,
    )


def _to_alert_log_item(group: SegmentGroup) -> AlertLogItem:
    segment = group.first_segment
    video = group.video
    return AlertLogItem(
        id=group.id,
        video_id=video.id,
        filename=video.filename,
        time=_format_clock_time(group.created_at),
        start_time=segment.start_time,
        end_time=group.last_segment.end_time,
        activity_type=group.activity_type,
        confidence_score=group.confidence_score,
        anomaly_score=group.anomaly_score,
        severity=_get_severity(group.anomaly_score),
        review_status=group.review_status,
        status=_get_display_status(group.review_status),
        created_at=group.created_at,
    )


def _to_critical_alert_item(group: SegmentGroup) -> CriticalAlertItem:
    segment = group.first_segment
    video = group.video
    return CriticalAlertItem(
        id=group.id,
        video_id=video.id,
        filename=video.filename,
        time=_format_clock_time(group.created_at),
        start_time=segment.start_time,
        end_time=group.last_segment.end_time,
        activity_type=group.activity_type,
        confidence_score=group.confidence_score,
        anomaly_score=group.anomaly_score,
        review_status=group.review_status,
        status=_get_display_status(group.review_status),
        created_at=group.created_at,
    )


def _get_severity(anomaly_score: float) -> str:
    if anomaly_score >= 0.85:
        return "HIGH"
    if anomaly_score >= 0.65:
        return "MEDIUM"
    return "LOW"


def _get_display_status(review_status: str) -> str:
    if review_status == "PENDING_REVIEW":
        return "Unreviewed"
    return "Reviewed"


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            return parsed.astimezone(VIETNAM_TIMEZONE).replace(tzinfo=None)
        return parsed
    except ValueError:
        return datetime.min


def _format_clock_time(value: str) -> str:
    parsed = _parse_timestamp(value)
    if parsed == datetime.min:
        return value

    today = vietnam_now().date()
    parsed_date = parsed.date()
    if parsed_date == today:
        return parsed.strftime("%H:%M")
    if parsed_date == today - timedelta(days=1):
        return f"Yesterday {parsed.strftime('%H:%M')}"
    if parsed_date.year == today.year:
        return parsed.strftime("%m/%d %H:%M")
    return parsed.strftime("%Y/%m/%d %H:%M")
