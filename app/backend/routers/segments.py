from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import ActivityLog, AnomalySegment, User, Video
from backend.schemas import AnomalySegmentRead, ApiResponse, FeedbackSubmitRequest
from backend.utils import vietnam_now_iso


router = APIRouter(prefix="/api/segments", tags=["segments"])
ADJACENT_SEGMENT_GAP_SECONDS = 1.0


def _api_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "message": message},
    )


@router.post("/{segment_id}/feedback", response_model=ApiResponse[AnomalySegmentRead])
def submit_feedback(
    segment_id: int,
    request: FeedbackSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[AnomalySegmentRead] | JSONResponse:
    segment = db.get(AnomalySegment, segment_id)
    if segment is None:
        return _api_error("Segment not found", status_code=404)

    other_description = _clean_optional_text(request.other_description)
    investigator_comment = _clean_optional_text(request.investigator_comment)
    if request.verified_label == "Other" and other_description is None:
        return _api_error("Other description is required when verified_label is Other")

    now = vietnam_now_iso()
    is_first_video_feedback = not _video_has_feedback(db, segment.video_id)
    grouped_segments = _get_adjacent_segment_group(db, segment)
    for grouped_segment in grouped_segments:
        grouped_segment.is_correct = 1 if request.is_correct else 0
        grouped_segment.verified_label = request.verified_label
        grouped_segment.other_description = (
            other_description if request.verified_label == "Other" else None
        )
        grouped_segment.investigator_comment = investigator_comment
        grouped_segment.feedback_submitted_at = now
        grouped_segment.review_status = _calculate_review_status(
            is_correct=request.is_correct,
            predicted_class=grouped_segment.predicted_class,
            verified_label=request.verified_label,
            investigator_comment=investigator_comment,
        )

    try:
        db.flush()
        video = db.get(Video, segment.video_id)
        if video is not None and is_first_video_feedback:
            db.add(
                ActivityLog(
                    type="REVIEW_COMPLETE",
                    title="Feedback submitted",
                    description=video.name,
                    video_id=video.id,
                    created_at=now,
                )
            )
        if video is not None and _all_segments_have_feedback(db, segment.video_id):
            video.status = "COMPLETED"
            video.updated_at = now
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        return _api_error(f"Feedback failed: {exc}", status_code=500)

    db.refresh(segment)
    return ApiResponse(
        success=True,
        data=AnomalySegmentRead.model_validate(segment),
        message="Feedback saved",
    )


def _get_adjacent_segment_group(
    db: Session,
    selected_segment: AnomalySegment,
) -> list[AnomalySegment]:
    segments = (
        db.query(AnomalySegment)
        .filter(AnomalySegment.video_id == selected_segment.video_id)
        .order_by(
            AnomalySegment.start_time.asc(),
            AnomalySegment.segment_index.asc(),
            AnomalySegment.id.asc(),
        )
        .all()
    )
    selected_index = next(
        index for index, segment in enumerate(segments) if segment.id == selected_segment.id
    )

    start_index = selected_index
    while start_index > 0 and _segments_should_group(
        segments[start_index - 1],
        segments[start_index],
    ):
        start_index -= 1

    end_index = selected_index
    while end_index + 1 < len(segments) and _segments_should_group(
        segments[end_index],
        segments[end_index + 1],
    ):
        end_index += 1

    return segments[start_index : end_index + 1]


def _segments_should_group(left: AnomalySegment, right: AnomalySegment) -> bool:
    return (
        left.predicted_class == right.predicted_class
        and right.start_time - left.end_time <= ADJACENT_SEGMENT_GAP_SECONDS
    )


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _calculate_review_status(
    is_correct: bool,
    predicted_class: str,
    verified_label: str,
    investigator_comment: str | None,
) -> str:
    if investigator_comment is not None:
        return "LOGGED"
    if not is_correct or verified_label != predicted_class:
        return "CORRECTED"
    return "LABEL_CORRECT"


def _all_segments_have_feedback(db: Session, video_id: str) -> bool:
    missing_feedback_count = (
        db.query(AnomalySegment)
        .filter(
            AnomalySegment.video_id == video_id,
            AnomalySegment.feedback_submitted_at.is_(None),
        )
        .count()
    )
    return missing_feedback_count == 0


def _video_has_feedback(db: Session, video_id: str) -> bool:
    feedback_count = (
        db.query(AnomalySegment)
        .filter(
            AnomalySegment.video_id == video_id,
            AnomalySegment.feedback_submitted_at.is_not(None),
        )
        .count()
    )
    return feedback_count > 0
