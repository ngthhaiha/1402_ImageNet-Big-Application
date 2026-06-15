from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


VideoStatus = Literal["WAITING", "PROCESSING", "PENDING_CONFIRM", "COMPLETED", "FAILED"]
ProgressStep = Literal[
    "WAITING",
    "PHASE1_START",
    "PHASE1_DONE",
    "PHASE2_DONE",
    "PENDING_CONFIRM",
    "FAILED",
]
JobStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
ReviewStatus = Literal["PENDING_REVIEW", "LABEL_CORRECT", "CORRECTED", "LOGGED"]
Severity = Literal["HIGH", "MEDIUM", "LOW"]
AlertDisplayStatus = Literal["Unreviewed", "Reviewed"]
InvestigationStatus = Literal["HIGH ALERT", "IN REVIEW", "VALIDATED"]
ActivityType = Literal["UPLOAD", "REVIEW_COMPLETE", "FLAG"]
NotificationType = Literal["success", "error", "warning", "info"]
AnomalyLabel = Literal[
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
]

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    success: bool
    data: DataT | None
    message: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class BatchBase(BaseModel):
    id: str
    name: str | None = None
    total_videos: int
    created_at: str


class BatchRead(BatchBase):
    model_config = ConfigDict(from_attributes=True)


class VideoBase(BaseModel):
    id: str
    batch_id: str | None = None
    filename: str
    name: str
    description: str | None = None
    location: str | None = None
    file_path: str
    file_size: int | None = None
    duration: float | None = None
    status: VideoStatus = "WAITING"
    progress_step: ProgressStep = "WAITING"
    error_message: str | None = None
    created_at: str
    updated_at: str


class VideoRead(VideoBase):
    model_config = ConfigDict(from_attributes=True)


class ProcessingJobBase(BaseModel):
    id: int
    video_id: str
    status: JobStatus = "PENDING"
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str


class ProcessingJobRead(ProcessingJobBase):
    model_config = ConfigDict(from_attributes=True)


class AnomalySegmentBase(BaseModel):
    id: int
    video_id: str
    segment_index: int
    start_time: float
    end_time: float
    anomaly_score: float
    predicted_class: AnomalyLabel
    confidence_score: float
    is_correct: int | None = None
    verified_label: AnomalyLabel | None = None
    other_description: str | None = None
    investigator_comment: str | None = None
    feedback_submitted_at: str | None = None
    review_status: ReviewStatus = "PENDING_REVIEW"
    created_at: str


class AnomalySegmentRead(AnomalySegmentBase):
    model_config = ConfigDict(from_attributes=True)


class VideoDetailRead(VideoRead):
    segments: list[AnomalySegmentRead] = Field(default_factory=list)


class UploadBatchRead(BaseModel):
    batch: BatchRead
    videos: list[VideoRead]


class VideoDurationProbeRead(BaseModel):
    filename: str
    duration: float


class BatchDetailRead(BatchRead):
    videos: list[VideoRead] = Field(default_factory=list)


class FeedbackSubmitRequest(BaseModel):
    is_correct: bool
    verified_label: AnomalyLabel
    other_description: str | None = None
    investigator_comment: str | None = None


class DashboardStatsRead(BaseModel):
    total_videos: int
    total_anomalies: int
    pending_reviews: int
    reviewed_cases: int


class DashboardDistributionRead(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: AnomalyLabel = Field(alias="class")
    count: int
    percentage: float


class DashboardAlertRead(BaseModel):
    id: int
    video_id: str
    time: str
    activity_type: AnomalyLabel
    confidence: float
    anomaly_score: float
    severity: Severity
    review_status: ReviewStatus
    is_correct: bool | None = None


class DashboardTopDetectionRead(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: AnomalyLabel = Field(alias="class")
    count: int


class DashboardInvestigationRead(BaseModel):
    video_id: str
    filename: str
    file_path: str
    duration: float | None = None
    file_size: int | None = None
    detected_activity: AnomalyLabel
    confidence: float
    anomaly_score: float
    investigation_status: InvestigationStatus
    created_at: str


class DashboardActivityRead(BaseModel):
    type: ActivityType
    title: str
    detail: str
    video_id: str | None = None
    created_at: str


class ProfileStatsRead(BaseModel):
    videos_uploaded: int
    cases_reviewed: int
    feedback_submitted: int


class ProfileActivityRead(BaseModel):
    id: int
    type: ActivityType
    title: str
    description: str | None = None
    video_id: str | None = None
    created_at: str


class NotificationItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: NotificationType
    title: str
    message: str
    target_url: str | None = None
    video_id: str | None = None
    is_read: bool
    created_at: str


class NotificationListResponse(BaseModel):
    items: list[NotificationItem]
    total: int


class UnreadCountResponse(BaseModel):
    count: int


class AlertStats(BaseModel):
    total_alerts: int
    high_severity: int
    pending_reviews: int
    reviewed_alerts: int


class AlertLogItem(BaseModel):
    id: int
    video_id: str
    filename: str
    time: str
    start_time: float
    end_time: float
    activity_type: AnomalyLabel
    confidence_score: float
    anomaly_score: float
    severity: Severity
    review_status: ReviewStatus
    status: AlertDisplayStatus
    created_at: str


class AlertLogResponse(BaseModel):
    items: list[AlertLogItem]
    total: int
    page: int
    total_pages: int


class DistributionItem(BaseModel):
    predicted_class: AnomalyLabel
    count: int
    percentage: float


class CriticalAlertItem(BaseModel):
    id: int
    video_id: str
    filename: str
    time: str
    start_time: float
    end_time: float
    activity_type: AnomalyLabel
    confidence_score: float
    anomaly_score: float
    review_status: ReviewStatus
    status: AlertDisplayStatus
    created_at: str
