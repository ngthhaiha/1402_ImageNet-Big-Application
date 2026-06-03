import axios from 'axios'
import type { AxiosProgressEvent } from 'axios'

import type {
  ApiResponse,
  BatchDetail,
  ActivityItem,
  AlertItem,
  DashboardFilter,
  DistributionItem,
  InvestigationItem,
  DashboardStats,
  ProfileActivityItem,
  ProfileStats,
  TopDetection,
  FeedbackSubmitRequest,
  AnomalySegment,
  UploadBatchResponse,
  UploadVideoMetadata,
  Video,
  VideoDetail,
  VideoDurationProbeResponse,
  VideoReport,
} from '../types/types'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
})

export function getUploadUrl(filePath: string): string {
  const normalizedPath = filePath.startsWith('/') ? filePath : `/${filePath}`
  return `${API_BASE_URL}${normalizedPath}`
}

export async function getVideos(): Promise<ApiResponse<Video[]>> {
  const response = await apiClient.get<ApiResponse<Video[]>>('/api/videos')
  return response.data
}

export async function getVideoDetail(videoId: string): Promise<ApiResponse<VideoDetail>> {
  const response = await apiClient.get<ApiResponse<VideoDetail>>(`/api/videos/${videoId}`)
  return response.data
}

export async function getBatchDetail(batchId: string): Promise<ApiResponse<BatchDetail>> {
  const response = await apiClient.get<ApiResponse<BatchDetail>>(`/api/batches/${batchId}`)
  return response.data
}

export async function getLatestBatch(): Promise<ApiResponse<BatchDetail>> {
  const response = await apiClient.get<ApiResponse<BatchDetail>>('/api/batches/latest')
  return response.data
}

export async function retryVideo(videoId: string): Promise<ApiResponse<Video>> {
  const response = await apiClient.post<ApiResponse<Video>>(`/api/videos/${videoId}/retry`)
  return response.data
}

export async function uploadVideos(
  files: File[],
  metadata: UploadVideoMetadata[] = [],
  onUploadProgress?: (event: AxiosProgressEvent) => void,
): Promise<ApiResponse<UploadBatchResponse>> {
  const formData = new FormData()
  const shouldAppendDurations =
    metadata.length === files.length &&
    metadata.every((itemMetadata) => itemMetadata.duration !== undefined)

  files.forEach((file, index) => {
    const itemMetadata = metadata[index]
    formData.append('files', file)
    if (itemMetadata?.name) {
      formData.append('names', itemMetadata.name)
    }
    if (itemMetadata?.description) {
      formData.append('descriptions', itemMetadata.description)
    }
    if (itemMetadata?.location) {
      formData.append('locations', itemMetadata.location)
    }
    if (shouldAppendDurations && itemMetadata?.duration !== undefined) {
      formData.append('durations', String(itemMetadata.duration))
    }
  })

  const response = await apiClient.post<ApiResponse<UploadBatchResponse>>(
    '/api/videos/upload',
    formData,
    { onUploadProgress },
  )
  return response.data
}

export async function probeVideoDuration(
  file: File,
): Promise<ApiResponse<VideoDurationProbeResponse>> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiClient.post<ApiResponse<VideoDurationProbeResponse>>(
    '/api/videos/probe-duration',
    formData,
  )
  return response.data
}

export async function submitFeedback(
  segmentId: number,
  payload: FeedbackSubmitRequest,
): Promise<ApiResponse<AnomalySegment>> {
  const response = await apiClient.post<ApiResponse<AnomalySegment>>(
    `/api/segments/${segmentId}/feedback`,
    payload,
  )
  return response.data
}

export interface DashboardFilterParams {
  anomaly_class?: string
  date_from?: string
  date_to?: string
}

function buildDashboardParams(
  filter?: DashboardFilter | DashboardFilterParams,
): Record<string, string> {
  const params: Record<string, string> = {}
  const anomalyClass = filter?.anomaly_class
  if (anomalyClass) {
    params.class = anomalyClass
  }
  if (filter?.date_from) {
    params.date_from = filter.date_from
  }
  if (filter?.date_to) {
    params.date_to = filter.date_to
  }
  return params
}

export async function getDashboardStats(): Promise<ApiResponse<DashboardStats>> {
  const response = await apiClient.get<ApiResponse<DashboardStats>>('/api/dashboard/stats')
  return response.data
}

export async function getDashboardDistribution(
  filter?: DashboardFilter,
): Promise<ApiResponse<DistributionItem[]>> {
  const response = await apiClient.get<ApiResponse<DistributionItem[]>>(
    '/api/dashboard/distribution',
    { params: buildDashboardParams(filter) },
  )
  return response.data
}

export async function getDashboardRecentAlerts(
  filter?: DashboardFilter,
  limit = 10,
): Promise<ApiResponse<AlertItem[]>> {
  const response = await apiClient.get<ApiResponse<AlertItem[]>>('/api/dashboard/recent-alerts', {
    params: { ...buildDashboardParams(filter), limit },
  })
  return response.data
}

export async function getDashboardTopDetections(
  filter?: DashboardFilter,
): Promise<ApiResponse<TopDetection[]>> {
  const response = await apiClient.get<ApiResponse<TopDetection[]>>(
    '/api/dashboard/top-detections',
    { params: buildDashboardParams(filter) },
  )
  return response.data
}

export async function getDashboardRecentInvestigations(
  filter?: DashboardFilter,
  limit = 5,
  offset = 0,
): Promise<ApiResponse<InvestigationItem[]>> {
  const response = await apiClient.get<ApiResponse<InvestigationItem[]>>(
    '/api/dashboard/recent-investigations',
    { params: { ...buildDashboardParams(filter), limit, offset } },
  )
  return response.data
}

export async function getDashboardRecentActivity(limit = 5): Promise<ApiResponse<ActivityItem[]>> {
  const response = await apiClient.get<ApiResponse<ActivityItem[]>>(
    '/api/dashboard/recent-activity',
    { params: { limit } },
  )
  return response.data
}

export async function getProfileStats(): Promise<ApiResponse<ProfileStats>> {
  const response = await apiClient.get<ApiResponse<ProfileStats>>('/api/profile/stats')
  return response.data
}

export async function getProfileActivity(
  limit = 10,
): Promise<ApiResponse<ProfileActivityItem[]>> {
  const response = await apiClient.get<ApiResponse<ProfileActivityItem[]>>(
    '/api/profile/activity',
    { params: { limit } },
  )
  return response.data
}

export async function getVideoReport(videoId: string): Promise<VideoReport> {
  const response = await apiClient.get<VideoReport>(`/api/videos/${videoId}/export`)
  return response.data
}

export async function downloadVideoReport(videoId: string): Promise<Blob> {
  const response = await apiClient.get<Blob>(`/api/videos/${videoId}/export`, {
    responseType: 'blob',
  })
  return response.data
}
