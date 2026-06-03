import type { JobStatus, ProgressStep, ReviewStatus, VideoStatus } from '../types/types'

type UploadStatus = 'Ready' | 'Invalid Format'

interface StatusBadgeProps {
  status: VideoStatus | ReviewStatus | JobStatus | ProgressStep | UploadStatus
}

const STATUS_CLASS_NAMES: Record<string, string> = {
  WAITING: 'bg-slate-100 text-slate-700 ring-slate-200',
  PROCESSING: 'bg-blue-100 text-blue-800 ring-blue-200',
  PENDING_CONFIRM: 'bg-amber-100 text-amber-800 ring-amber-200',
  COMPLETED: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
  FAILED: 'bg-red-100 text-red-800 ring-red-200',
  PENDING: 'bg-slate-100 text-slate-700 ring-slate-200',
  RUNNING: 'bg-blue-100 text-blue-800 ring-blue-200',
  PENDING_REVIEW: 'bg-slate-100 text-slate-700 ring-slate-200',
  LABEL_CORRECT: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
  CORRECTED: 'bg-orange-100 text-orange-800 ring-orange-200',
  LOGGED: 'bg-indigo-100 text-indigo-800 ring-indigo-200',
  PHASE1_START: 'bg-blue-100 text-blue-800 ring-blue-200',
  PHASE1_DONE: 'bg-blue-100 text-blue-800 ring-blue-200',
  PHASE2_DONE: 'bg-blue-100 text-blue-800 ring-blue-200',
  Ready: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
  'Invalid Format': 'bg-red-100 text-red-800 ring-red-200',
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const className = STATUS_CLASS_NAMES[status] ?? 'bg-slate-100 text-slate-700 ring-slate-200'

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${className}`}
    >
      {status.replaceAll('_', ' ')}
    </span>
  )
}
