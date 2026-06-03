import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Archive, Download, XCircle } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'

import { downloadVideoReport, getVideoDetail } from '../api/api'
import { AnomalyTimeline } from '../components/AnomalyTimeline'
import { FeedbackPanel } from '../components/FeedbackPanel'
import { InvestigationPanel } from '../components/InvestigationPanel'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { SegmentsTable } from '../components/SegmentsTable'
import { VideoPlayer } from '../components/VideoPlayer'
import type {
  AnomalySegment,
  ProgressStep,
  Video,
  VideoDetail as VideoDetailData,
} from '../types/types'

const PROGRESS_STEP_PERCENT: Record<ProgressStep, number> = {
  WAITING: 0,
  PHASE1_START: 10,
  PHASE1_DONE: 50,
  PHASE2_DONE: 90,
  PENDING_CONFIRM: 100,
  FAILED: 0,
}

const PROGRESS_STEP_LABEL: Record<ProgressStep, string> = {
  WAITING: 'Waiting for processing...',
  PHASE1_START: 'Running Phase 1: Anomaly Detection...',
  PHASE1_DONE: 'Phase 1 Complete. Running Phase 2: Classification...',
  PHASE2_DONE: 'Phase 2 Complete. Saving results...',
  PENDING_CONFIRM: 'Analysis Complete',
  FAILED: 'Processing failed',
}

const PROGRESS_WIDTH_CLASS: Record<number, string> = {
  0: 'w-0',
  10: 'w-1/10',
  50: 'w-1/2',
  90: 'w-9/10',
  100: 'w-full',
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unable to load video detail'
}

function getProgressWidthClass(progressStep: ProgressStep): string {
  return PROGRESS_WIDTH_CLASS[PROGRESS_STEP_PERCENT[progressStep]]
}

function ProcessingProgress({ progressStep }: { progressStep: ProgressStep }) {
  const percent = PROGRESS_STEP_PERCENT[progressStep]
  const isComplete = progressStep === 'PENDING_CONFIRM'

  return (
    <section className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
      <div className="mb-2 flex items-center justify-between text-sm font-semibold text-[#131B2E]">
        <span>Processing Progress</span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#E2E5F0]">
        <div
          className={`h-full rounded-full ${isComplete ? 'bg-emerald-500' : 'bg-[#004AC6]'} ${getProgressWidthClass(progressStep)}`}
        />
      </div>
      <p className="mt-2 text-sm font-medium text-[#434655]">{PROGRESS_STEP_LABEL[progressStep]}</p>
    </section>
  )
}

export function VideoDetail() {
  const { id } = useParams<{ id: string }>()
  const [video, setVideo] = useState<Video | null>(null)
  const [segments, setSegments] = useState<AnomalySegment[]>([])
  const [selectedSegmentId, setSelectedSegmentId] = useState<number | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const selectedSegment = useMemo(
    () => segments.find((segment) => segment.id === selectedSegmentId) ?? null,
    [segments, selectedSegmentId],
  )
  const investigationSegment = useMemo(
    () => (selectedSegment && video ? { ...selectedSegment, video_name: video.name } : null),
    [selectedSegment, video],
  )

  const loadVideo = useCallback(async () => {
    if (!id) {
      setError('Video ID is required')
      setIsLoading(false)
      return
    }

    try {
      const response = await getVideoDetail(id)
      if (!response.success || response.data === null) {
        throw new Error(response.message)
      }

      const detail: VideoDetailData = response.data
      setVideo(detail)
      setSegments(detail.segments)
      setSelectedSegmentId((currentSelectedId) => {
        if (detail.segments.length === 0) {
          return null
        }

        const selectedStillExists =
          currentSelectedId !== null &&
          detail.segments.some((segment) => segment.id === currentSelectedId)

        return selectedStillExists ? currentSelectedId : detail.segments[0].id
      })
      setError(null)
    } catch (loadError) {
      setError(getErrorMessage(loadError))
    } finally {
      setIsLoading(false)
    }
  }, [id])

  useEffect(() => {
    void loadVideo()
  }, [loadVideo])

  useEffect(() => {
    if (video?.status !== 'PROCESSING') {
      return undefined
    }

    const intervalId = window.setInterval(() => {
      void loadVideo()
    }, 3000)

    return () => window.clearInterval(intervalId)
  }, [loadVideo, video?.status])

  function handleFeedbackSubmitted(updatedSegment: AnomalySegment) {
    setSegments((currentSegments) => {
      const nextSegments = currentSegments.map((segment) =>
        segment.id === updatedSegment.id ? updatedSegment : segment,
      )
      const allSegmentsHaveFeedback =
        nextSegments.length > 0 &&
        nextSegments.every((segment) => segment.feedback_submitted_at !== null)

      if (allSegmentsHaveFeedback) {
        setVideo((currentVideo) =>
          currentVideo ? { ...currentVideo, status: 'COMPLETED' } : currentVideo,
        )
      }

      return nextSegments
    })
    setSelectedSegmentId(updatedSegment.id)
  }

  async function handleExportReport() {
    if (!id || video?.status === 'WAITING') {
      return
    }

    setIsExporting(true)
    setError(null)
    try {
      const blob = await downloadVideoReport(id)
      const downloadUrl = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = `report_${id}.json`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(downloadUrl)
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : 'Export report failed')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <section className="min-h-screen bg-[#FAF8FF] px-8 py-8 text-[#131B2E]">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm font-medium text-[#737686]">
              <Link to="/queue" className="text-[#737686]">
                Analysis Queue
              </Link>{' '}
              / <span className="font-semibold text-[#004AC6]">Segment Review</span>
            </p>
            <h2 className="mt-2 text-3xl font-semibold text-[#131B2E]">Video Investigation</h2>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-[#737686] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100"
            >
              <Archive className="h-4 w-4" aria-hidden="true" />
              Archive Case
            </button>
            <button
              type="button"
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#004AC6] px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => void handleExportReport()}
              disabled={!video || video.status === 'WAITING' || isExporting}
            >
              <Download className="h-4 w-4" aria-hidden="true" />
              {isExporting ? 'Exporting...' : 'Export Report'}
            </button>
          </div>
        </header>

        {error ? (
          <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <XCircle className="h-5 w-5 shrink-0" aria-hidden="true" />
            {error}
          </div>
        ) : null}

        {isLoading ? (
          <div className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
            <LoadingSpinner label="Loading video detail" />
          </div>
        ) : video ? (
          <div className="flex flex-col gap-6 lg:flex-row">
            <div className="flex flex-col gap-6 lg:w-[65%]">
              <VideoPlayer
                video={video}
                selectedSegment={selectedSegment}
                currentTime={currentTime}
                setCurrentTime={setCurrentTime}
                videoRef={videoRef}
              />

              {video.status === 'PROCESSING' ? (
                <ProcessingProgress progressStep={video.progress_step} />
              ) : null}

              <AnomalyTimeline
                video={video}
                segments={segments}
                selectedSegmentId={selectedSegmentId}
                setSelectedSegmentId={setSelectedSegmentId}
                videoRef={videoRef}
              />

              <SegmentsTable
                video={video}
                segments={segments}
                selectedSegmentId={selectedSegmentId}
                setSelectedSegmentId={setSelectedSegmentId}
                videoRef={videoRef}
              />
            </div>

            <aside className="flex flex-col gap-6 lg:w-[35%]">
              <InvestigationPanel segment={investigationSegment} />
              <FeedbackPanel segment={selectedSegment} onFeedbackSubmitted={handleFeedbackSubmitted} />
            </aside>
          </div>
        ) : null}
      </div>
    </section>
  )
}
