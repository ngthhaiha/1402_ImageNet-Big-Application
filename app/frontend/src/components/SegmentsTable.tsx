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

interface SegmentGroup {
  key: string
  segments: AnomalySegment[]
  start_time: number
  end_time: number
  predicted_class: AnomalyLabel
  confidence_score: number
  review_status: ReviewStatus | 'MIXED'
}

const ADJACENT_SEGMENT_GAP_SECONDS = 1

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

function getAverageConfidence(segments: AnomalySegment[]): number {
  const total = segments.reduce((sum, segment) => sum + segment.confidence_score, 0)
  return total / segments.length
}

function getGroupReviewStatus(segments: AnomalySegment[]): SegmentGroup['review_status'] {
  const firstStatus = segments[0].review_status
  const allSameStatus = segments.every((segment) => segment.review_status === firstStatus)
  return allSameStatus ? firstStatus : 'MIXED'
}

function toSegmentGroup(groupSegments: AnomalySegment[]): SegmentGroup {
  const firstSegment = groupSegments[0]
  const lastSegment = groupSegments[groupSegments.length - 1]

  return {
    key: groupSegments.map((segment) => segment.id).join('-'),
    segments: groupSegments,
    start_time: firstSegment.start_time,
    end_time: lastSegment.end_time,
    predicted_class: firstSegment.predicted_class,
    confidence_score: getAverageConfidence(groupSegments),
    review_status: getGroupReviewStatus(groupSegments),
  }
}

function groupAdjacentSegments(segments: AnomalySegment[]): SegmentGroup[] {
  const sortedSegments = [...segments].sort((left, right) => {
    if (left.start_time !== right.start_time) {
      return left.start_time - right.start_time
    }

    return left.segment_index - right.segment_index
  })
  const groups: AnomalySegment[][] = []

  sortedSegments.forEach((segment) => {
    const lastGroup = groups[groups.length - 1]
    const lastSegment = lastGroup?.[lastGroup.length - 1]
    const isSameActivity = lastSegment?.predicted_class === segment.predicted_class
    const isTimeAdjacent =
      lastSegment !== undefined &&
      segment.start_time - lastSegment.end_time <= ADJACENT_SEGMENT_GAP_SECONDS

    if (lastSegment && isSameActivity && isTimeAdjacent) {
      lastGroup.push(segment)
      return
    }

    groups.push([segment])
  })

  return groups.map(toSegmentGroup)
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

function GroupReviewStatusBadge({ status }: { status: SegmentGroup['review_status'] }) {
  if (status !== 'MIXED') {
    return <ReviewStatusBadge status={status} />
  }

  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-[#F2F3FF] px-2.5 py-1 text-xs font-medium text-[#434655]">
      <span className="h-2 w-2 rounded-full bg-[#004AC6]" aria-hidden="true" />
      Mixed Review
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

  const segmentGroups = groupAdjacentSegments(segments)

  function seekToSegment(segment: AnomalySegment) {
    setSelectedSegmentId(segment.id)
    if (videoRef.current) {
      videoRef.current.currentTime = segment.start_time
    }
  }

  function seekToGroup(group: SegmentGroup) {
    seekToSegment(group.segments[0])
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
            {segmentGroups.length > 0 ? (
              segmentGroups.map((group) => (
                <tr
                  key={group.key}
                  id={`segment-row-${group.segments[0].id}`}
                  className={[
                    'cursor-pointer border-b border-[#C3C6D7] transition hover:bg-slate-100 last:border-b-0',
                    selectedSegmentId !== null &&
                    group.segments.some((segment) => segment.id === selectedSegmentId)
                      ? 'bg-[#F2F3FF]'
                      : '',
                  ].join(' ')}
                  onClick={() => seekToGroup(group)}
                >
                  <td className="px-6 py-4 text-sm font-medium text-[#131B2E]">
                    {group.segments.slice(1).map((segment) => (
                      <span
                        key={segment.id}
                        id={`segment-row-${segment.id}`}
                        style={{ display: 'inline-block', width: 0, height: 0, overflow: 'hidden' }}
                        aria-hidden="true"
                      />
                    ))}
                    {formatTime(group.start_time)} - {formatTime(group.end_time)}
                  </td>
                  <td className="px-6 py-4">
                    <ActivityBadge label={group.predicted_class} />
                  </td>
                  <td className="px-6 py-4 text-sm font-bold text-[#131B2E]">
                    {formatConfidence(group.confidence_score)}
                  </td>
                  <td className="px-6 py-4">
                    <GroupReviewStatusBadge status={group.review_status} />
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
