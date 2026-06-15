from dataclasses import dataclass
from typing import Iterable

from backend.models import AnomalySegment, Video


ADJACENT_SEGMENT_GAP_SECONDS = 1.0


@dataclass
class SegmentGroup:
    video: Video
    segments: list[AnomalySegment]

    @property
    def first_segment(self) -> AnomalySegment:
        return self.segments[0]

    @property
    def last_segment(self) -> AnomalySegment:
        return self.segments[-1]

    @property
    def id(self) -> int:
        return self.first_segment.id

    @property
    def activity_type(self) -> str:
        return self.first_segment.predicted_class

    @property
    def confidence_score(self) -> float:
        return sum(segment.confidence_score for segment in self.segments) / len(self.segments)

    @property
    def anomaly_score(self) -> float:
        return max(segment.anomaly_score for segment in self.segments)

    @property
    def review_status(self) -> str:
        if any(segment.review_status == "PENDING_REVIEW" for segment in self.segments):
            return "PENDING_REVIEW"
        return self.first_segment.review_status

    @property
    def is_correct(self) -> bool | None:
        values = {segment.is_correct for segment in self.segments}
        if len(values) == 1:
            value = values.pop()
            return None if value is None else bool(value)
        return None

    @property
    def created_at(self) -> str:
        return max(segment.created_at for segment in self.segments)

    @property
    def sort_id(self) -> int:
        return max(segment.id for segment in self.segments)


def group_segment_rows(
    rows: Iterable[tuple[AnomalySegment, Video]],
) -> list[SegmentGroup]:
    rows_by_video: dict[str, list[tuple[AnomalySegment, Video]]] = {}
    for segment, video in rows:
        rows_by_video.setdefault(video.id, []).append((segment, video))

    groups: list[SegmentGroup] = []
    for video_rows in rows_by_video.values():
        sorted_rows = sorted(
            video_rows,
            key=lambda row: (row[0].start_time, row[0].segment_index, row[0].id),
        )
        current_segments: list[AnomalySegment] = []
        current_video: Video | None = None

        for segment, video in sorted_rows:
            last_segment = current_segments[-1] if current_segments else None
            is_same_activity = (
                last_segment is not None
                and last_segment.predicted_class == segment.predicted_class
            )
            is_time_adjacent = (
                last_segment is not None
                and segment.start_time - last_segment.end_time
                <= ADJACENT_SEGMENT_GAP_SECONDS
            )

            if current_segments and (not is_same_activity or not is_time_adjacent):
                groups.append(SegmentGroup(video=current_video or video, segments=current_segments))
                current_segments = []

            current_video = video
            current_segments.append(segment)

        if current_segments:
            groups.append(SegmentGroup(video=current_video or sorted_rows[0][1], segments=current_segments))

    return groups
