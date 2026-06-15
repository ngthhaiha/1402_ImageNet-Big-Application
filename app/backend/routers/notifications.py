from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import Notification, User
from backend.schemas import (
    ApiResponse,
    NotificationItem,
    NotificationListResponse,
    UnreadCountResponse,
)


router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _api_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "message": message},
    )


@router.get("", response_model=ApiResponse[NotificationListResponse])
def get_notifications(
    is_read: bool | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[NotificationListResponse]:
    query = db.query(Notification)
    if is_read is not None:
        query = query.filter(Notification.is_read == int(is_read))

    total = query.with_entities(func.count(Notification.id)).scalar() or 0
    notifications = (
        query.order_by(Notification.created_at.desc(), Notification.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return ApiResponse(
        success=True,
        data=NotificationListResponse(
            items=[
                NotificationItem.model_validate(notification)
                for notification in notifications
            ],
            total=total,
        ),
        message="Notifications loaded",
    )


@router.get("/unread-count", response_model=ApiResponse[UnreadCountResponse])
def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[UnreadCountResponse]:
    count = (
        db.query(func.count(Notification.id))
        .filter(Notification.is_read == 0)
        .scalar()
        or 0
    )
    return ApiResponse(
        success=True,
        data=UnreadCountResponse(count=count),
        message="Unread notification count loaded",
    )


@router.patch("/read-all", response_model=ApiResponse[dict[str, int]])
def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[dict[str, int]]:
    notifications = db.query(Notification).filter(Notification.is_read == 0).all()
    updated = len(notifications)
    for notification in notifications:
        notification.is_read = 1

    db.commit()
    return ApiResponse(
        success=True,
        data={"updated": updated},
        message="All notifications marked as read",
    )


@router.patch("/{notification_id}/read", response_model=ApiResponse[NotificationItem])
def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiResponse[NotificationItem] | JSONResponse:
    notification = db.get(Notification, notification_id)
    if notification is None:
        return _api_error("Notification not found", status_code=404)

    notification.is_read = 1
    db.commit()
    db.refresh(notification)
    return ApiResponse(
        success=True,
        data=NotificationItem.model_validate(notification),
        message="Notification marked as read",
    )
