import sqlite3
import sys
from pathlib import Path
from threading import Lock

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils import create_notification, vietnam_now_iso

try:
    from ai_pipeline import run_phase1, run_phase2
except ModuleNotFoundError as exc:
    if exc.name != "ai_pipeline":
        raise
    from backend.ai_pipeline import run_phase1, run_phase2


PENDING_JOB_QUERY = """
SELECT *
FROM processing_jobs
WHERE status = 'PENDING'
ORDER BY created_at ASC
LIMIT 1
"""
ANOMALY_LABELS = {
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
    "Normal",
    "Other",
}
ADJACENT_SEGMENT_GAP_SECONDS = 1.0

_worker_lock = Lock()
_is_running = False





def start_worker_loop(db_path: str) -> None:
    global _is_running

    with _worker_lock:
        if _is_running:
            return
        _is_running = True

    try:
        while True:
            job = _poll_next_job(db_path)
            if job is None:
                break

            run_pipeline(job["video_id"], db_path)
    finally:
        with _worker_lock:
            _is_running = False


def run_pipeline(video_id: str, db_path: str) -> None:
    connection = _connect(db_path)
    try:
        video = connection.execute(
            "SELECT * FROM videos WHERE id = ?",
            (video_id,),
        ).fetchone()
        if video is None:
            raise ValueError(f"Video not found: {video_id}")

        _mark_running(connection, video_id)

        video_path = _resolve_video_path(video["file_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video['file_path']}")

        phase1_segments = run_phase1(str(video_path))
        db_segments = _normalize_phase1_segments(phase1_segments)
        _update_progress(connection, video_id, "PHASE1_DONE")

        classified_segments = []
        for segment, db_segment in zip(phase1_segments, db_segments):
            result = _normalize_phase2_result(run_phase2(str(video_path), segment))
            classified_segments.append((db_segment, result))
        _update_progress(connection, video_id, "PHASE2_DONE")

        now = vietnam_now_iso()
        for segment_index, (segment, result) in enumerate(classified_segments):
            connection.execute(
                """
                INSERT INTO anomaly_segments (
                    video_id,
                    segment_index,
                    start_time,
                    end_time,
                    anomaly_score,
                    predicted_class,
                    confidence_score,
                    review_status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING_REVIEW', ?)
                """,
                (
                    video_id,
                    segment_index,
                    segment["start_time"],
                    segment["end_time"],
                    segment["anomaly_score"],
                    result["predicted_class"],
                    result["confidence_score"],
                    now,
                ),
            )
        flagged_segments = [
            segment for segment, _ in classified_segments if segment["anomaly_score"] > 0.8
        ]
        if flagged_segments:
            connection.execute(
                """
                INSERT INTO activity_log (
                    type,
                    title,
                    description,
                    video_id,
                    created_at
                )
                VALUES ('FLAG', 'High anomaly flagged', ?, ?, ?)
                """,
                (
                    f"{len(flagged_segments)} high-risk segment(s) detected in {video['name']}",
                    video_id,
                    now,
                ),
            )

        _create_video_notifications(connection, video, video_id, classified_segments)

        connection.execute(
            """
            UPDATE videos
            SET status = 'PENDING_CONFIRM',
                progress_step = 'PENDING_CONFIRM',
                error_message = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (now, video_id),
        )
        connection.execute(
            """
            UPDATE processing_jobs
            SET status = 'COMPLETED',
                finished_at = ?
            WHERE video_id = ?
            """,
            (now, video_id),
        )
        connection.commit()
        _maybe_create_batch_complete_notification(connection, video_id)
    except Exception as exc:
        connection.rollback()
        _mark_failed(connection, video_id, str(exc))
    finally:
        connection.close()


def _connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=TRUNCATE")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.row_factory = sqlite3.Row
    return connection


def _poll_next_job(db_path: str) -> sqlite3.Row | None:
    connection = _connect(db_path)
    try:
        return connection.execute(PENDING_JOB_QUERY).fetchone()
    finally:
        connection.close()


def _mark_running(connection: sqlite3.Connection, video_id: str) -> None:
    now = vietnam_now_iso()
    connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'RUNNING',
            started_at = ?,
            finished_at = NULL
        WHERE video_id = ?
        """,
        (now, video_id),
    )
    connection.execute(
        """
        UPDATE videos
        SET status = 'PROCESSING',
            progress_step = 'PHASE1_START',
            error_message = NULL,
            updated_at = ?
        WHERE id = ?
        """,
        (now, video_id),
    )
    connection.commit()


def _update_progress(connection: sqlite3.Connection, video_id: str, progress_step: str) -> None:
    now = vietnam_now_iso()
    connection.execute(
        """
        UPDATE videos
        SET progress_step = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (progress_step, now, video_id),
    )
    connection.commit()


def _mark_failed(connection: sqlite3.Connection, video_id: str, error_message: str) -> None:
    now = vietnam_now_iso()
    connection.execute(
        """
        UPDATE videos
        SET status = 'FAILED',
            progress_step = 'FAILED',
            error_message = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (error_message, now, video_id),
    )
    connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'FAILED',
            finished_at = ?
        WHERE video_id = ?
        """,
        (now, video_id),
    )
    connection.commit()
    video = connection.execute(
        "SELECT * FROM videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    video_name = video["filename"] if video is not None else video_id
    create_notification(
        connection,
        notification_type="error",
        title="Video processing failed",
        message=f"Video {video_name} failed during processing. {error_message}",
        target_url=f"/videos/{video_id}",
        video_id=video_id,
    )
    _maybe_create_batch_complete_notification(connection, video_id)


def _create_video_notifications(
    connection: sqlite3.Connection,
    video: sqlite3.Row,
    video_id: str,
    classified_segments: list[tuple[dict, dict]],
) -> None:
    video_name = video["filename"]
    if classified_segments:
        group_count = _count_classified_segment_groups(classified_segments)
        create_notification(
            connection,
            notification_type="success",
            title="Video detected as abnormal",
            message=_format_anomaly_segment_message(video_name, group_count),
            target_url=f"/videos/{video_id}",
            video_id=video_id,
        )

        low_confidence_segments = [
            result
            for _, result in classified_segments
            if result["confidence_score"] < 0.6
        ]
        if low_confidence_segments:
            create_notification(
                connection,
                notification_type="warning",
                title="Low confidence detection",
                message=(
                    f"Video {video_name} has {len(low_confidence_segments)} "
                    "segment(s) with low confidence. Manual review recommended."
                ),
                target_url=f"/videos/{video_id}",
                video_id=video_id,
            )
    else:
        create_notification(
            connection,
            notification_type="info",
            title="Video processing complete",
            message=f"Video {video_name} processed with no anomaly detected.",
            target_url=f"/videos/{video_id}",
            video_id=video_id,
        )


def _count_classified_segment_groups(classified_segments: list[tuple[dict, dict]]) -> int:
    group_count = 0
    previous_segment = None
    previous_class = None

    sorted_segments = sorted(
        classified_segments,
        key=lambda item: item[0]["start_time"],
    )

    for segment, result in sorted_segments:
        predicted_class = result["predicted_class"]
        is_same_activity = predicted_class == previous_class
        is_time_adjacent = (
            previous_segment is not None
            and segment["start_time"] - previous_segment["end_time"]
            <= ADJACENT_SEGMENT_GAP_SECONDS
        )

        if not is_same_activity or not is_time_adjacent:
            group_count += 1

        previous_segment = segment
        previous_class = predicted_class

    return group_count


def _format_anomaly_segment_message(video_name: str, group_count: int) -> str:
    if group_count == 1:
        return f"Video {video_name} has an anomaly segment waiting for review."

    return f"Video {video_name} has {group_count} anomaly segments waiting for review."


def _maybe_create_batch_complete_notification(
    connection: sqlite3.Connection,
    video_id: str,
) -> None:
    video = connection.execute(
        "SELECT batch_id FROM videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    if video is None or video["batch_id"] is None:
        return

    batch_id = video["batch_id"]
    videos = connection.execute(
        "SELECT status FROM videos WHERE batch_id = ?",
        (batch_id,),
    ).fetchall()
    if not videos:
        return

    terminal_statuses = {"PENDING_CONFIRM", "FAILED", "COMPLETED"}
    if any(video_row["status"] not in terminal_statuses for video_row in videos):
        return

    success_count = sum(
        1
        for video_row in videos
        if video_row["status"] in {"PENDING_CONFIRM", "COMPLETED"}
    )
    total_count = len(videos)
    create_notification(
        connection,
        notification_type="info",
        title="Batch processing complete",
        message=f"{success_count} of {total_count} videos processed successfully.",
        target_url="/queue",
        video_id=None,
    )


def _resolve_video_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    return BACKEND_DIR / path


def _normalize_phase1_segments(segments: list[dict]) -> list[dict]:
    if not isinstance(segments, list):
        raise ValueError("Phase 1 result must be a list")

    normalized_segments = []
    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        anomaly_score = float(segment["anomaly_score"])

        if start_time < 0:
            raise ValueError("Segment start_time must be >= 0")
        if end_time <= start_time:
            raise ValueError("Segment end_time must be greater than start_time")
        if not 0 <= anomaly_score <= 1:
            raise ValueError("Segment anomaly_score must be between 0 and 1")

        normalized_segments.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "anomaly_score": anomaly_score,
            }
        )

    return normalized_segments


def _normalize_phase2_result(result: dict) -> dict:
    predicted_class = str(result["predicted_class"])
    confidence_score = float(result["confidence_score"])

    if predicted_class not in ANOMALY_LABELS:
        raise ValueError(f"Invalid predicted_class: {predicted_class}")
    if not 0 <= confidence_score <= 1:
        raise ValueError("confidence_score must be between 0 and 1")

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
    }
