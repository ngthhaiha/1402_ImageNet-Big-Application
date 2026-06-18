export const ANOMALY_LABELS = [
  'Abuse',
  'Arrest',
  'Arson',
  'Assault',
  'Burglary',
  'Explosion',
  'Fighting',
  'RoadAccidents',
  'Robbery',
  'Shooting',
  'Shoplifting',
  'Stealing',
  'Vandalism',
  'Normal',
  'Other',
] as const

export type AnomalyLabel = (typeof ANOMALY_LABELS)[number]

export type VideoStatus =
  | 'WAITING'
  | 'PROCESSING'
  | 'PENDING_CONFIRM'
  | 'COMPLETED'
  | 'FAILED'

export type ProgressStep =
  | 'WAITING'
  | 'PHASE1_START'
  | 'PHASE1_DONE'
  | 'PHASE2_DONE'
  | 'PENDING_CONFIRM'
  | 'FAILED'

export type JobStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED'

export type ReviewStatus = 'PENDING_REVIEW' | 'LABEL_CORRECT' | 'CORRECTED' | 'LOGGED'
export type Severity = 'HIGH' | 'MEDIUM' | 'LOW'
export type DisplayReviewStatus =
  | 'Unreviewed'
  | 'Validated'
  | 'Corrected'
  | 'Logged'
  | 'False Positive'
export type InvestigationStatus = DisplayReviewStatus
export type DashboardActivityType = 'UPLOAD' | 'REVIEW_COMPLETE' | 'FLAG'
export type NotificationType = 'success' | 'error' | 'warning' | 'info'

export const CLASS_COLORS: Record<string, string> = {
  Abuse: '#E91E63',
  Arrest: '#00BCD4',
  Arson: '#FF5722',
  Assault: '#795548',
  Burglary: '#9C27B0',
  Explosion: '#EF4444',
  Fighting: '#BA1A1A',
  RoadAccidents: '#4CAF50',
  Robbery: '#004AC6',
  Shooting: '#F59E0B',
  Shoplifting: '#8B5CF6',
  Stealing: '#607D8B',
  Vandalism: '#FF9800',
  Normal: '#4CAF50',
  Other: '#9E9E9E',
}

export const TYPE_COLORS: Record<string, string> = {
  success: '#059669',
  error: '#BA1A1A',
  warning: '#D97706',
  info: '#004AC6',
}

export interface ApiResponse<T> {
  success: boolean
  data: T | null
  message: string
}

export interface User {
  id: number
  username: string
  email: string
}

export interface AuthResponse {
  token: string
  user: User
}

export interface Notification {
  id: number
  type: NotificationType
  title: string
  message: string
  target_url: string | null
  video_id: string | null
  is_read: boolean
  created_at: string
}

export interface NotificationListResponse {
  items: Notification[]
  total: number
}

export interface Batch {
  id: string
  name: string | null
  total_videos: number
  created_at: string
}

export interface Video {
  id: string
  batch_id: string | null
  filename: string
  name: string
  description: string | null
  location: string | null
  file_path: string
  file_size: number | null
  duration: number | null
  status: VideoStatus
  progress_step: ProgressStep
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface ProcessingJob {
  id: number
  video_id: string
  status: JobStatus
  started_at: string | null
  finished_at: string | null
  created_at: string
}

export interface AnomalySegment {
  id: number
  video_id: string
  segment_index: number
  start_time: number
  end_time: number
  anomaly_score: number
  predicted_class: AnomalyLabel
  confidence_score: number
  is_correct: number | null
  verified_label: AnomalyLabel | null
  other_description: string | null
  investigator_comment: string | null
  feedback_submitted_at: string | null
  review_status: ReviewStatus
  created_at: string
}

export interface VideoDetail extends Video {
  segments: AnomalySegment[]
}

export interface BatchDetail extends Batch {
  videos: Video[]
}

export interface UploadBatchResponse {
  batch: Batch
  videos: Video[]
}

export interface VideoDurationProbeResponse {
  filename: string
  duration: number
}

export interface UploadVideoMetadata {
  name?: string
  description?: string
  location?: string
  duration?: number
}

export interface FeedbackSubmitRequest {
  is_correct: boolean
  verified_label: AnomalyLabel
  other_description?: string | null
  investigator_comment?: string | null
}

export interface DashboardStats {
  total_videos: number
  total_anomalies: number
  pending_reviews: number
  reviewed_cases: number
}

export interface AlertStats {
  total_alerts: number
  high_severity: number
  pending_reviews: number
  reviewed_alerts: number
}

export interface AlertFilter {
  name: string
  activity: string
  severity: '' | Severity
  status: '' | 'PENDING_REVIEW' | 'LABEL_CORRECT' | 'CORRECTED' | 'LOGGED' | 'FALSE_POSITIVE'
  date: string
}

export interface AlertLogItem {
  id: number
  video_id: string
  filename: string
  time: string
  start_time: number
  end_time: number
  activity_type: AnomalyLabel
  confidence_score: number
  anomaly_score: number
  severity: Severity
  review_status: ReviewStatus
  is_correct: boolean | null
  verified_label: AnomalyLabel | null
  status: DisplayReviewStatus
  created_at: string
}

export interface AlertLogResponse {
  items: AlertLogItem[]
  total: number
  page: number
  total_pages: number
}

export interface CriticalAlertItem {
  id: number
  video_id: string
  filename: string
  time: string
  start_time: number
  end_time: number
  activity_type: AnomalyLabel
  confidence_score: number
  anomaly_score: number
  review_status: ReviewStatus
  is_correct: boolean | null
  verified_label: AnomalyLabel | null
  status: DisplayReviewStatus
  created_at: string
}

export interface DistributionItem {
  class: string
  predicted_class?: AnomalyLabel
  count: number
  percentage: number
}

export interface AlertItem {
  id: number
  video_id: string
  time: string
  activity_type: string
  confidence: number
  anomaly_score: number
  severity: Severity
  review_status: ReviewStatus | 'PROCESSING'
  is_correct: boolean | null
  verified_label: AnomalyLabel | null
}

export interface TopDetection {
  class: string
  count: number
}

export interface InvestigationItem {
  video_id: string
  filename: string
  file_path: string
  duration: number | null
  file_size: number | null
  detected_activity: string
  confidence: number
  anomaly_score: number
  investigation_status: InvestigationStatus
  created_at: string
}

export interface ActivityItem {
  type: DashboardActivityType
  title: string
  detail: string
  video_id: string | null
  created_at: string
}

export interface ProfileStats {
  videos_uploaded: number
  cases_reviewed: number
  feedback_submitted: number
}

export interface ProfileActivityItem {
  id: number
  type: DashboardActivityType
  title: string
  description: string | null
  video_id: string | null
  created_at: string
}

export interface DashboardFilter {
  anomaly_class: string
  date_from: string
  date_to: string
}

export interface VideoReportSegment {
  segment_id: string
  time_range: string
  predicted_class: AnomalyLabel
  confidence_score: number
  anomaly_score: number
  review_status: ReviewStatus
  is_correct: boolean | null
  verified_label: AnomalyLabel | null
  other_description: string | null
  investigator_comment: string | null
  feedback_submitted_at: string | null
}

export interface VideoReport {
  video: {
    id: string
    name: string
    location: string | null
    duration: number | null
    status: VideoStatus
    created_at: string
  }
  summary: {
    total_segments: number
    total_anomalies: number
    feedback_submitted: number
    pending_review: number
  }
  segments: VideoReportSegment[]
}
