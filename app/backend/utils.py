from datetime import datetime, timedelta, timezone
from threading import Lock


VIETNAM_TIMEZONE = timezone(timedelta(hours=7))

_video_id_lock = Lock()
_last_video_second = ""
_video_sequence = 0


def vietnam_now() -> datetime:
    return datetime.now(VIETNAM_TIMEZONE)


def vietnam_now_iso() -> str:
    return vietnam_now().isoformat()


def generate_video_id() -> str:
    global _last_video_second, _video_sequence

    timestamp = vietnam_now().strftime("%Y%m%d_%H%M%S")
    with _video_id_lock:
        if timestamp != _last_video_second:
            _last_video_second = timestamp
            _video_sequence = 1
        else:
            _video_sequence += 1

        return f"{timestamp}_{_video_sequence:04d}"


def generate_batch_id() -> str:
    timestamp = vietnam_now().strftime("%Y%m%d_%H%M%S")
    return f"BCH-{timestamp}"


def format_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


def create_notification(
    db,
    notification_type: str,
    title: str,
    message: str,
    target_url: str | None = None,
    video_id: str | None = None,
):
    created_at = datetime.utcnow().isoformat()

    if hasattr(db, "add"):
        from backend.models import Notification

        notification = Notification(
            type=notification_type,
            title=title,
            message=message,
            target_url=target_url,
            video_id=video_id,
            is_read=0,
            created_at=created_at,
        )
        db.add(notification)
        db.commit()
        return notification

    db.execute(
        """
        INSERT INTO notifications (
            type,
            title,
            message,
            target_url,
            video_id,
            is_read,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, 0, ?)
        """,
        (notification_type, title, message, target_url, video_id, created_at),
    )
    db.commit()
    return None
