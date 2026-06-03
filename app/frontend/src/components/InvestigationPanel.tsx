import type { AnomalySegment, ReviewStatus } from '../types/types'

interface InvestigationPanelSegment extends AnomalySegment {
  video_name: string
}

interface InvestigationPanelProps {
  segment: InvestigationPanelSegment | null
}

const REVIEW_STATUS_LABEL: Record<ReviewStatus, string> = {
  PENDING_REVIEW: 'Pending Review',
  LABEL_CORRECT: 'Label Correct',
  CORRECTED: 'Corrected',
  LOGGED: 'Logged',
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

function getPercentClass(score: number): string {
  const percent = Math.min(100, Math.max(0, Math.round(score * 100)))
  return `w-pct-${percent}`
}

export function InvestigationPanel({ segment }: InvestigationPanelProps) {
  return (
    <section className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3 border-b border-[#C3C6D7] pb-4">
        <h3 className="text-lg font-semibold text-[#131B2E]">Investigation Summary</h3>
        <span className="rounded bg-[#DBE1FF] px-2 py-1 text-xs font-bold uppercase text-[#004AC6]">
          Verified AI
        </span>
      </div>

      {segment ? (
        <div className="flex flex-col gap-4 text-sm">
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Video Name</span>
            <span className="text-right font-semibold text-[#131B2E]">{segment.video_name}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Segment ID</span>
            <span className="font-semibold text-[#004AC6]">#SEG-{segment.segment_index + 1}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Predicted Activity</span>
            <span className="inline-flex items-center gap-2 font-bold text-[#BA1A1A]">
              <span className="h-2 w-2 rounded-full bg-[#BA1A1A]" aria-hidden="true" />
              {segment.predicted_class}
            </span>
          </div>
          <div>
            <div className="mb-1 flex items-center justify-between gap-4">
              <span className="text-xs font-medium text-[#737686]">Confidence</span>
              <span className="text-base font-bold text-[#131B2E]">
                {formatConfidence(segment.confidence_score)}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[#E2E5F0]">
              <div className={`h-full rounded-full bg-[#004AC6] ${getPercentClass(segment.confidence_score)}`} />
            </div>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Timestamp</span>
            <span className="font-semibold text-[#131B2E]">
              {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
            </span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Status</span>
            <span className="rounded-full bg-[#F2F3FF] px-2.5 py-1 text-xs font-medium text-[#434655]">
              {REVIEW_STATUS_LABEL[segment.review_status]}
            </span>
          </div>
        </div>
      ) : (
        <p className="text-sm text-[#434655]">Select a segment to inspect the AI result.</p>
      )}
    </section>
  )
}
