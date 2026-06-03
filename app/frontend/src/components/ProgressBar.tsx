import type { ProgressStep } from '../types/types'

interface ProgressBarProps {
  value?: number
  progressStep?: ProgressStep
  label?: string
}

const PROGRESS_STEP_PERCENT: Record<ProgressStep, number> = {
  WAITING: 0,
  PHASE1_START: 10,
  PHASE1_DONE: 50,
  PHASE2_DONE: 90,
  PENDING_CONFIRM: 100,
  FAILED: 0,
}

export function ProgressBar({ value, progressStep, label }: ProgressBarProps) {
  const rawPercent = value ?? (progressStep ? PROGRESS_STEP_PERCENT[progressStep] : 0)
  const percent = Math.min(100, Math.max(0, rawPercent))
  const isFailed = progressStep === 'FAILED'
  const isComplete = percent === 100 && !isFailed
  const barColor = isFailed
    ? 'bg-red-500'
    : isComplete
      ? 'bg-emerald-500'
      : 'bg-blue-600'

  return (
    <div className="w-full">
      <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
        <span>{label ?? progressStep?.replaceAll('_', ' ') ?? 'Progress'}</span>
        <span>{isFailed ? 'Failed' : `${percent}%`}</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}
