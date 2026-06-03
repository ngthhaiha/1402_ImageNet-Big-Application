import type { Dispatch, RefObject, SetStateAction } from 'react'
import { Download, Filter } from 'lucide-react'

import type { AnomalyLabel, AnomalySegment, ReviewStatus, Video } from '../types/types'

interface SegmentsTableProps {
  video: Video
  segments: AnomalySegment[]
  selectedSegmentId: number | null
  setSelectedSegmentId: Dispatch<SetStateAction<number | null>>
  videoRef: RefObject<HTMLVideoElement | null>
}

const REVIEW_STATUS_META: Record<ReviewStatus, { label: string; dotClassName: string }> = {
  PENDING_REVIEW: { label: 'Pending Review', dotClassName: 'bg-[#737686]' },
  LABEL_CORRECT: { label: 'Label Correct', dotClassName: 'bg-emerald-500' },
  CORRECTED: { label: 'Corrected', dotClassName: 'bg-orange-500' },
  LOGGED: { label: 'Logged', dotClassName: 'bg-[#BA1A1A]' },
}

function formatTime(seconds: number): string {
  const rounded = Math.max(0, Math.round(seconds))
  const minutes = Math.floor(rounded / 60)
  const remainingSeconds = rounded % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
}

function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

function ActivityBadge({ label }: { label: AnomalyLabel }) {
  const className =
    label === 'Normal'
      ? 'bg-[#F2F3FF] text-[#434655]'
      : 'bg-[#FEE2E2] text-[#BA1A1A]'

  return (
    <span className={`inline-flex rounded px-2 py-1 text-xs font-bold uppercase ${className}`}>
      {label}
    </span>
  )
}

function ReviewStatusBadge({ status }: { status: ReviewStatus }) {
  const meta = REVIEW_STATUS_META[status]
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-[#F2F3FF] px-2.5 py-1 text-xs font-medium text-[#434655]">
      <span className={`h-2 w-2 rounded-full ${meta.dotClassName}`} aria-hidden="true" />
      {meta.label}
    </span>
  )
}

export function SegmentsTable({
  video,
  segments,
  selectedSegmentId,
  setSelectedSegmentId,
  videoRef,
}: SegmentsTableProps) {
  if (video.status !== 'PENDING_CONFIRM' && video.status !== 'COMPLETED') {
    return null
  }

  function seekToSegment(segment: AnomalySegment) {
    setSelectedSegmentId(segment.id)
    if (videoRef.current) {
      videoRef.current.currentTime = segment.start_time
    }
  }

  return (
    <section className="overflow-hidden rounded-xl border border-[#C3C6D7] bg-white shadow-sm">
      <div className="flex items-center justify-between gap-3 border-b border-[#C3C6D7] px-6 py-4">
        <h3 className="text-base font-semibold text-[#131B2E]">Detected Segments</h3>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded border border-[#C3C6D7] bg-white text-[#434655] transition hover:bg-slate-100"
            aria-label="Filter segments"
          >
            <Filter className="h-4 w-4" aria-hidden="true" />
          </button>
          <button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded border border-[#C3C6D7] bg-white text-[#434655] transition hover:bg-slate-100"
            aria-label="Download segments"
          >
            <Download className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse">
          <thead>
            <tr className="border-b border-[#C3C6D7] text-left text-xs font-semibold uppercase tracking-wide text-[#737686]">
              <th className="px-6 py-4">Time Range</th>
              <th className="px-6 py-4">Predicted Activity</th>
              <th className="px-6 py-4">Confidence</th>
              <th className="px-6 py-4">Review Status</th>
            </tr>
          </thead>
          <tbody>
            {segments.length > 0 ? (
              segments.map((segment) => (
                <tr
                  key={segment.id}
                  className={[
                    'cursor-pointer border-b border-[#C3C6D7] transition hover:bg-slate-100 last:border-b-0',
                    selectedSegmentId === segment.id ? 'bg-[#F2F3FF]' : '',
                  ].join(' ')}
                  onClick={() => seekToSegment(segment)}
                >
                  <td className="px-6 py-4 text-sm font-medium text-[#131B2E]">
                    {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
                  </td>
                  <td className="px-6 py-4">
                    <ActivityBadge label={segment.predicted_class} />
                  </td>
                  <td className="px-6 py-4 text-sm font-bold text-[#131B2E]">
                    {formatConfidence(segment.confidence_score)}
                  </td>
                  <td className="px-6 py-4">
                    <ReviewStatusBadge status={segment.review_status} />
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-6 py-8 text-center text-sm text-[#434655]" colSpan={4}>
                  No detected segments.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}
