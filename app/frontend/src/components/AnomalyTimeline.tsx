import type { Dispatch, RefObject, SetStateAction } from 'react'
import { Activity } from 'lucide-react'

import type { AnomalySegment, Video } from '../types/types'

interface AnomalyTimelineProps {
  video: Video
  segments: AnomalySegment[]
  selectedSegmentId: number | null
  setSelectedSegmentId: Dispatch<SetStateAction<number | null>>
  videoRef: RefObject<HTMLVideoElement | null>
}

function formatTime(seconds: number): string {
  const rounded = Math.max(0, Math.round(seconds))
  const minutes = Math.floor(rounded / 60)
  const remainingSeconds = rounded % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
}

function getTimelineDuration(video: Video, segments: AnomalySegment[]): number {
  if (video.duration !== null && Number.isFinite(video.duration) && video.duration > 0) {
    return video.duration
  }

  return Math.max(...segments.map((segment) => segment.end_time), 1)
}

function clampPercent(percent: number): number {
  return Math.min(100, Math.max(0, Math.round(percent)))
}

function getBlockPosition(segment: AnomalySegment, duration: number) {
  const left = clampPercent((segment.start_time / duration) * 100)
  const rawWidth = clampPercent(((segment.end_time - segment.start_time) / duration) * 100)
  const width = Math.min(100 - left, Math.max(1, rawWidth))
  return { left, width }
}

export function AnomalyTimeline({
  video,
  segments,
  selectedSegmentId,
  setSelectedSegmentId,
  videoRef,
}: AnomalyTimelineProps) {
  if (video.status !== 'PENDING_CONFIRM' && video.status !== 'COMPLETED') {
    return null
  }

  const duration = getTimelineDuration(video, segments)
  const markers = Array.from({ length: 6 }, (_, index) => (duration / 5) * index)

  function seekToSegment(segment: AnomalySegment) {
    setSelectedSegmentId(segment.id)
    if (videoRef.current) {
      videoRef.current.currentTime = segment.start_time
    }
  }

  return (
    <section className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-[#004AC6]" aria-hidden="true" />
          <h3 className="text-sm font-semibold text-[#131B2E]">Analysis Timeline</h3>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs text-[#737686]">
          <span className="inline-flex items-center gap-2">
            <span className="h-4 w-4 rounded border border-[#C3C6D7] bg-[#C3C6D7]" aria-hidden="true" />
            Normal
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="h-4 w-4 rounded bg-[#BA1A1A]" aria-hidden="true" />
            Detected Anomaly
          </span>
        </div>
      </div>

      <div className="relative h-6 overflow-hidden rounded bg-[#E2E5F0]">
        {segments.map((segment) => {
          const position = getBlockPosition(segment, duration)
          return (
            <button
              key={segment.id}
              type="button"
              className={[
                'absolute top-0 h-full border-0 bg-[#BA1A1A] p-0 transition hover:opacity-90',
                `left-pct-${position.left}`,
                `w-pct-${position.width}`,
                selectedSegmentId === segment.id ? 'ring-2 ring-current' : '',
              ].join(' ')}
              onClick={() => seekToSegment(segment)}
              aria-label={`Seek to anomaly segment ${segment.segment_index + 1}`}
            />
          )
        })}
      </div>

      <div className="mt-2 flex items-center justify-between text-xs font-medium text-[#737686]">
        {markers.map((marker) => (
          <span key={marker}>{formatTime(marker)}</span>
        ))}
      </div>
    </section>
  )
}
