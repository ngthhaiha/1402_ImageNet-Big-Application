import sqlite3
from pathlib import Path
from threading import Lock

from backend.utils import vietnam_now_iso


BACKEND_DIR = Path(__file__).resolve().parent
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

_worker_lock = Lock()
_is_running = False


def run_phase1(video_path: str) -> list[dict]:
    """
    Returns list of segments: [{start_time, end_time, anomaly_score}]
    REPLACE THIS with actual pipeline import when ready.
    """
    return [
        {"start_time": 10.5, "end_time": 25.0, "anomaly_score": 0.87},
        {"start_time": 78.2, "end_time": 91.4, "anomaly_score": 0.73},
    ]


def run_phase2(video_path: str, segment: dict) -> dict:
    """
    Returns: {predicted_class, confidence_score}
    REPLACE THIS with actual pipeline import when ready.
    """
    return {"predicted_class": "Fighting", "confidence_score": 0.91}


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

        phase1_segments = _normalize_phase1_segments(run_phase1(str(video_path)))
        _update_progress(connection, video_id, "PHASE1_DONE")

        classified_segments = []
        for segment in phase1_segments:
            result = _normalize_phase2_result(run_phase2(str(video_path), segment))
            classified_segments.append((segment, result))
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
    except Exception as exc:
        connection.rollback()
        _mark_failed(connection, video_id, str(exc))
    finally:
        connection.close()


def _connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
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
