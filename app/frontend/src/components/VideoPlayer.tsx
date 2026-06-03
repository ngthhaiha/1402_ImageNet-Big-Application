import type { RefObject, SyntheticEvent } from 'react'

import { getUploadUrl } from '../api/api'
import type { AnomalySegment, Video } from '../types/types'

interface VideoPlayerProps {
  video: Video
  selectedSegment: AnomalySegment | null
  currentTime: number
  setCurrentTime: (currentTime: number) => void
  videoRef: RefObject<HTMLVideoElement | null>
}

function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

export function VideoPlayer({
  video,
  selectedSegment,
  currentTime,
  setCurrentTime,
  videoRef,
}: VideoPlayerProps) {
  const isSelectedSegmentActive =
    selectedSegment !== null &&
    currentTime >= selectedSegment.start_time &&
    currentTime <= selectedSegment.end_time

  function handleTimeUpdate(event: SyntheticEvent<HTMLVideoElement>) {
    setCurrentTime(event.currentTarget.currentTime)
  }

  return (
    <section className="relative h-[366px] overflow-hidden rounded-xl border border-[#C3C6D7] bg-black shadow-sm">
      {isSelectedSegmentActive ? (
        <div className="absolute left-4 top-4 z-20 inline-flex items-center gap-2 rounded-full bg-[#BA1A1A] px-3 py-1 text-xs font-bold uppercase text-white">
          <span className="h-2 w-2 rounded-full bg-white" aria-hidden="true" />
          ANOMALY DETECTED: {selectedSegment.predicted_class.toUpperCase()} (
          {formatConfidence(selectedSegment.confidence_score)})
        </div>
      ) : null}

      <video
        ref={videoRef}
        className="h-full w-full bg-black object-cover"
        controls
        preload="metadata"
        src={getUploadUrl(video.file_path)}
        onTimeUpdate={handleTimeUpdate}
      >
        <track kind="captions" />
      </video>
    </section>
  )
}
