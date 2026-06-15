from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import ActivityLog, AnomalySegment, User, Video
from backend.schemas import ApiResponse, ProfileActivityRead, ProfileStatsRead


router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("/stats", response_model=ApiResponse[ProfileStatsRead])
def get_profile_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[ProfileStatsRead]:
    feedback_count = (
        db.query(AnomalySegment)
        .filter(AnomalySegment.feedback_submitted_at.is_not(None))
        .count()
    )
    stats = ProfileStatsRead(
        videos_uploaded=db.query(Video).count(),
        cases_reviewed=feedback_count,
        feedback_submitted=feedback_count,
    )
    return ApiResponse(success=True, data=stats, message="Profile stats loaded")


@router.get("/activity", response_model=ApiResponse[list[ProfileActivityRead]])
def get_profile_activity(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[list[ProfileActivityRead]]:
    activities = (
        db.query(ActivityLog)
        .order_by(ActivityLog.created_at.desc(), ActivityLog.id.desc())
        .limit(limit)
        .all()
    )
    return ApiResponse(
        success=True,
        data=[
            ProfileActivityRead(
                id=activity.id,
                type=activity.type,
                title=activity.title,
                description=activity.description,
                video_id=activity.video_id,
                created_at=activity.created_at,
            )
            for activity in activities
        ],
        message="Profile activity loaded",
    )
