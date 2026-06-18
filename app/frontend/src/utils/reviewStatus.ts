import type { AnomalyLabel, ReviewStatus } from '../types/types'

export type DisplayReviewStatus =
  | 'Unreviewed'
  | 'Validated'
  | 'Corrected'
  | 'Logged'
  | 'False Positive'

export interface ReviewStatusDisplay {
  label: DisplayReviewStatus
  badgeClass: string
  dotClassName: string
}

export function getReviewStatusDisplay(
  reviewStatus: ReviewStatus | 'PROCESSING',
  isCorrect?: boolean | number | null,
  verifiedLabel?: AnomalyLabel | null,
): ReviewStatusDisplay {
  if (reviewStatus === 'PENDING_REVIEW') {
    return {
      label: 'Unreviewed',
      badgeClass: 'alert-status-unreviewed',
      dotClassName: 'bg-[#737686]',
    }
  }

  if (reviewStatus === 'LABEL_CORRECT') {
    return {
      label: 'Validated',
      badgeClass: 'alert-status-validated',
      dotClassName: 'bg-emerald-500',
    }
  }

  if (reviewStatus === 'LOGGED') {
    return {
      label: 'Logged',
      badgeClass: 'alert-status-primary',
      dotClassName: 'bg-[#BA1A1A]',
    }
  }

  if (reviewStatus === 'CORRECTED' && verifiedLabel !== 'Normal') {
    return {
      label: 'Corrected',
      badgeClass: 'alert-status-primary',
      dotClassName: 'bg-orange-500',
    }
  }

  if (isCorrect === false || isCorrect === 0 || verifiedLabel === 'Normal') {
    return {
      label: 'False Positive',
      badgeClass: 'alert-status-muted',
      dotClassName: 'bg-[#505F76]',
    }
  }

  return {
    label: 'Unreviewed',
    badgeClass: 'alert-status-unreviewed',
    dotClassName: 'bg-[#737686]',
  }
}
