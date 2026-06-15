import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Download,
  FileVideo,
  ListVideo,
  RefreshCw,
  RotateCcw,
  XCircle,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'

import { getBatchDetail, getLatestBatch, retryVideo } from '../api/api'
import { PageHeader } from '../components/PageHeader'
import { StatusBadge } from '../components/StatusBadge'
import type { BatchDetail, ProgressStep, Video, VideoStatus } from '../types/types'

const ROWS_PER_PAGE = 10

const PROGRESS_STEP_PERCENT: Record<ProgressStep, number> = {
  WAITING: 0,
  PHASE1_START: 10,
  PHASE1_DONE: 50,
  PHASE2_DONE: 90,
  PENDING_CONFIRM: 100,
  FAILED: 0,
}

const PROGRESS_WIDTH_CLASS: Record<number, string> = {
  0: 'w-0',
  10: 'w-1/10',
  50: 'w-1/2',
  90: 'w-9/10',
  100: 'w-full',
}

type SummaryKey = 'DONE' | 'ACTIVE' | 'QUEUED' | 'ERRORS'

interface SummaryCard {
  key: SummaryKey
  label: string
  value: number
  icon: LucideIcon
  badgeClassName: string
  accentClassName: string
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unable to load queue'
}

function formatDateTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  return `${date.toLocaleDateString()} • ${date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })}`
}

function formatTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return '--:--:--'
  }

  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return '--:--:--'
  }

  const rounded = Math.round(seconds)
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const remainingSeconds = rounded % 60
  return [hours, minutes, remainingSeconds].map((part) => String(part).padStart(2, '0')).join(':')
}

function getProgressPercent(video: Video): number {
  if (video.status === 'COMPLETED') {
    return 100
  }

  if (video.status === 'FAILED') {
    return 0
  }

  return PROGRESS_STEP_PERCENT[video.progress_step]
}

function getProgressWidthClass(percent: number): string {
  return PROGRESS_WIDTH_CLASS[percent] ?? 'w-0'
}

function getProgressFillClass(status: VideoStatus): string {
  if (status === 'FAILED') {
    return 'bg-red-500'
  }

  return 'bg-[#004AC6]'
}

function isPollingTerminal(video: Video): boolean {
  return video.status === 'COMPLETED' || video.status === 'FAILED'
}

function renderAction(
  video: Video,
  retryingVideoId: string | null,
  onRetry: (videoId: string) => void,
) {
  if (video.status === 'WAITING') {
    return <span className="text-sm font-medium text-[#434655] opacity-70">—</span>
  }

  if (video.status === 'PROCESSING') {
    return (
      <Link
        to={`/videos/${video.id}`}
        className="inline-flex items-center justify-center rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#004AC6] transition hover:bg-slate-100"
      >
        Monitor
      </Link>
    )
  }

  if (video.status === 'PENDING_CONFIRM') {
    return (
      <Link
        to={`/videos/${video.id}`}
        className="inline-flex items-center justify-center rounded-lg bg-[#004AC6] px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:opacity-90"
      >
        Review
      </Link>
    )
  }

  if (video.status === 'COMPLETED') {
    return (
      <Link to={`/videos/${video.id}`} className="text-sm font-bold text-[#004AC6]">
        View Detail
      </Link>
    )
  }

  return (
    <button
      type="button"
      className="inline-flex items-center justify-center gap-2 rounded-lg border border-red-200 bg-white px-4 py-2 text-sm font-semibold text-red-800 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60"
      onClick={() => onRetry(video.id)}
      disabled={retryingVideoId === video.id}
    >
      <RotateCcw className="h-4 w-4" aria-hidden="true" />
      {retryingVideoId === video.id ? 'Retrying' : 'Retry'}
    </button>
  )
}

export function Queue() {
  const [searchParams] = useSearchParams()
  const batchIdParam = searchParams.get('batch_id')
  const [batch, setBatch] = useState<BatchDetail | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [retryingVideoId, setRetryingVideoId] = useState<string | null>(null)

  const videos = useMemo(() => batch?.videos ?? [], [batch])
  const totalVideos = videos.length
  const doneCount = videos.filter(
    (video) => video.status === 'PENDING_CONFIRM' || video.status === 'COMPLETED',
  ).length
  const activeCount = videos.filter((video) => video.status === 'PROCESSING').length
  const queuedCount = videos.filter((video) => video.status === 'WAITING').length
  const errorCount = videos.filter((video) => video.status === 'FAILED').length
  const completionPercent = totalVideos > 0 ? Math.round((doneCount / totalVideos) * 100) : 0
  const completionWidthClass = getProgressWidthClass(
    completionPercent >= 100 ? 100 : completionPercent >= 90 ? 90 : completionPercent >= 50 ? 50 : completionPercent >= 10 ? 10 : 0,
  )
  const allPollingTerminal = totalVideos > 0 && videos.every(isPollingTerminal)
  const totalPages = Math.max(1, Math.ceil(totalVideos / ROWS_PER_PAGE))
  const currentPage = Math.min(page, totalPages)
  const pageStartIndex = (currentPage - 1) * ROWS_PER_PAGE
  const pageEndIndex = Math.min(pageStartIndex + ROWS_PER_PAGE, totalVideos)
  const visibleVideos = videos.slice(pageStartIndex, pageEndIndex)

  const summaryCards = useMemo<SummaryCard[]>(
    () => [
      {
        key: 'DONE',
        label: 'Videos processed',
        value: doneCount,
        icon: CheckCircle2,
        badgeClassName: 'bg-[#DBE1FF] text-[#004AC6]',
        accentClassName: '',
      },
      {
        key: 'ACTIVE',
        label: 'Currently running',
        value: activeCount,
        icon: Activity,
        badgeClassName: 'bg-[#DBE1FF] text-blue-800',
        accentClassName: 'border-l-4 border-[#C3C6D7]',
      },
      {
        key: 'QUEUED',
        label: 'Waiting in line',
        value: queuedCount,
        icon: Clock3,
        badgeClassName: 'bg-slate-100 text-[#434655]',
        accentClassName: '',
      },
      {
        key: 'ERRORS',
        label: 'Need retry',
        value: errorCount,
        icon: AlertTriangle,
        badgeClassName: 'bg-red-100 text-red-800',
        accentClassName: '',
      },
    ],
    [activeCount, doneCount, errorCount, queuedCount],
  )

  const loadBatch = useCallback(
    async (showRefreshing = false) => {
      if (showRefreshing) {
        setIsRefreshing(true)
      } else {
        setIsLoading(true)
      }
      setError(null)

      try {
        const response = batchIdParam
          ? await getBatchDetail(batchIdParam)
          : await getLatestBatch()

        if (!response.success || response.data === null) {
          throw new Error(response.message)
        }

        const batchData = response.data
        setBatch(batchData)
        setPage((current) =>
          Math.min(current, Math.max(1, Math.ceil(batchData.videos.length / ROWS_PER_PAGE))),
        )
      } catch (loadError) {
        setError(getErrorMessage(loadError))
      } finally {
        setIsLoading(false)
        setIsRefreshing(false)
      }
    },
    [batchIdParam],
  )

  useEffect(() => {
    void loadBatch()
  }, [loadBatch])

  useEffect(() => {
    if (allPollingTerminal) {
      return undefined
    }

    const intervalId = window.setInterval(() => {
      void loadBatch(true)
    }, 5000)

    return () => window.clearInterval(intervalId)
  }, [allPollingTerminal, loadBatch])

  async function handleRetry(videoId: string) {
    setRetryingVideoId(videoId)
    setError(null)

    try {
      const response = await retryVideo(videoId)
      if (!response.success || response.data === null) {
        throw new Error(response.message)
      }
      await loadBatch(true)
    } catch (retryError) {
      setError(getErrorMessage(retryError))
    } finally {
      setRetryingVideoId(null)
    }
  }

  function goToPage(nextPage: number) {
    setPage(Math.min(totalPages, Math.max(1, nextPage)))
  }

  return (
    <section className="min-h-screen bg-[#FAF8FF] px-8 py-8 text-[#131B2E]">
      <PageHeader pageName="Analysis Queue" />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <header>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-3xl font-semibold text-[#131B2E]">Analysis Queue</h2>
              <p className="mt-2 text-base text-[#434655]">
                Real-time AI telemetry for the active investigation batch.
              </p>
            </div>

            <button
              type="button"
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-[#737686] bg-white px-6 py-4 text-base font-semibold text-[#131B2E] transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => void loadBatch(true)}
              disabled={isRefreshing}
            >
              <RefreshCw className={isRefreshing ? 'h-5 w-5 animate-spin' : 'h-5 w-5'} aria-hidden="true" />
              Refresh Queue
            </button>
          </div>
        </header>

        {error ? (
          <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <XCircle className="h-5 w-5 shrink-0" aria-hidden="true" />
            {error}
          </div>
        ) : null}

        <section className="rounded-xl border border-[#C3C6D7] bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="flex min-w-0 items-start gap-4">
              <ListVideo className="h-8 w-8 shrink-0 text-[#004AC6]" aria-hidden="true" />
              <div className="min-w-0">
                <div className="mb-2 flex flex-wrap items-center gap-3">
                  <span className="rounded bg-[#DBE1FF] px-2 py-1 text-xs font-bold text-[#004AC6]">
                    ACTIVE BATCH
                  </span>
                  <span className="truncate text-sm font-medium text-[#434655]">
                    {batch?.id ?? 'No batch loaded'}
                  </span>
                </div>
                <h3 className="truncate text-xl font-semibold text-[#131B2E]">
                  {batch?.name ?? (isLoading ? 'Loading batch...' : 'No active batch')}
                </h3>
                <p className="mt-2 text-sm text-[#434655]">
                  Uploaded: {batch ? formatDateTime(batch.created_at) : '--'}
                </p>
              </div>
            </div>

            <div className="text-left lg:text-right">
              <p className="text-3xl font-bold text-[#131B2E]">{completionPercent}%</p>
              <p className="mt-1 text-sm font-medium text-[#434655]">Completion</p>
            </div>
          </div>

          <div className="mt-8">
            <div className="mb-2 flex items-center justify-between text-sm font-medium text-[#434655]">
              <span>
                {doneCount} of {totalVideos} Videos Processed
              </span>
              <span>{allPollingTerminal ? 'Complete' : 'Processing...'}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-[#E2E5F0]">
              <div className={`h-full rounded-full bg-[#004AC6] ${completionWidthClass}`} />
            </div>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {summaryCards.map((card) => {
            const Icon = card.icon
            return (
              <article
                key={card.key}
                className={`rounded-xl border border-[#C3C6D7] bg-white p-4 shadow-sm ${card.accentClassName}`}
              >
                <div className="mb-8 flex items-center justify-between gap-3">
                  <Icon className="h-6 w-6 text-[#004AC6]" aria-hidden="true" />
                  <span className={`rounded px-2 py-1 text-xs font-bold ${card.badgeClassName}`}>
                    {card.key}
                  </span>
                </div>
                <p className="text-3xl font-bold text-[#131B2E]">{card.value}</p>
                <p className="mt-1 text-xs font-medium uppercase tracking-wide text-[#434655]">
                  {card.label}
                </p>
              </article>
            )
          })}
        </section>

        <section className="overflow-hidden rounded-xl border border-[#C3C6D7] bg-white shadow-sm">
          <div className="flex flex-col gap-3 border-b border-[#C3C6D7] px-6 py-4 lg:flex-row lg:items-center lg:justify-between">
            <h3 className="text-lg font-semibold text-[#131B2E]">Queue Details</h3>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition disabled:cursor-not-allowed disabled:opacity-50"
                disabled
              >
                <Download className="h-4 w-4" aria-hidden="true" />
                Download Report
              </button>
              <button
                type="button"
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition disabled:cursor-not-allowed disabled:opacity-50"
                disabled
              >
                <XCircle className="h-4 w-4" aria-hidden="true" />
                Cancel All
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] border-collapse">
              <thead>
                <tr className="border-b border-[#C3C6D7] text-left text-xs font-semibold uppercase tracking-wide text-[#434655]">
                  <th className="px-6 py-4">Video Name</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4">Progress</th>
                  <th className="px-6 py-4">Duration</th>
                  <th className="px-6 py-4">Submitted</th>
                  <th className="px-6 py-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {visibleVideos.length > 0 ? (
                  visibleVideos.map((video) => {
                    const progressPercent = getProgressPercent(video)
                    return (
                      <tr
                        key={video.id}
                        className="border-b border-[#C3C6D7] transition hover:bg-slate-100 last:border-b-0"
                      >
                        <td className="px-6 py-4">
                          <div className="flex min-w-0 items-center gap-3">
                            <FileVideo className="h-6 w-6 shrink-0 text-[#004AC6]" aria-hidden="true" />
                            <div className="min-w-0">
                              <p className="truncate text-sm font-semibold text-[#131B2E]">{video.name}</p>
                              <p className="truncate text-xs text-[#434655]">{video.filename}</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <StatusBadge status={video.status} />
                        </td>
                        <td className="px-6 py-4">
                          {video.status === 'FAILED' ? (
                            <p className="max-w-2xl truncate text-sm font-medium text-red-800">
                              Error: {video.error_message ?? 'Processing failed'}
                            </p>
                          ) : (
                            <div className="min-w-40">
                              <div className="mb-1 flex items-center justify-between text-xs font-medium text-[#434655]">
                                <span>{video.progress_step.replaceAll('_', ' ')}</span>
                                <span>{progressPercent}%</span>
                              </div>
                              <div className="h-2 overflow-hidden rounded-full bg-[#E2E5F0]">
                                <div
                                  className={`h-full rounded-full ${getProgressFillClass(video.status)} ${getProgressWidthClass(progressPercent)}`}
                                />
                              </div>
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4 text-sm text-[#434655]">
                          {formatDuration(video.duration)}
                        </td>
                        <td className="px-6 py-4 text-sm text-[#434655]">{formatTime(video.created_at)}</td>
                        <td className="px-6 py-4 text-right">
                          {renderAction(video, retryingVideoId, handleRetry)}
                        </td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td className="px-6 py-12 text-center text-sm text-[#434655]" colSpan={6}>
                      {isLoading ? 'Loading queue...' : 'No videos in the latest batch.'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex flex-col gap-3 border-t border-[#C3C6D7] px-6 py-4 lg:flex-row lg:items-center lg:justify-between">
            <p className="text-sm text-[#434655]">
              Showing {totalVideos === 0 ? 0 : pageStartIndex + 1} to {pageEndIndex} of {totalVideos} results
            </p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded border border-[#C3C6D7] bg-white text-[#434655] transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" aria-hidden="true" />
              </button>
              {Array.from({ length: totalPages }, (_, index) => index + 1).map((pageNumber) => (
                <button
                  key={pageNumber}
                  type="button"
                  className={
                    pageNumber === currentPage
                      ? 'inline-flex h-8 w-8 items-center justify-center rounded bg-[#004AC6] text-sm font-semibold text-white'
                      : 'inline-flex h-8 w-8 items-center justify-center rounded border border-[#C3C6D7] bg-white text-sm font-semibold text-[#434655] transition hover:bg-slate-100'
                  }
                  onClick={() => goToPage(pageNumber)}
                >
                  {pageNumber}
                </button>
              ))}
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded border border-[#C3C6D7] bg-white text-[#434655] transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                aria-label="Next page"
              >
                <ChevronRight className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
          </div>
        </section>
      </div>
    </section>
  )
}
