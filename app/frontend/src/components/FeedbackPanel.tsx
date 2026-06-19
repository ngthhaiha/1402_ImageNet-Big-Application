import { useEffect, useState } from 'react'
import { Check, Pencil, X } from 'lucide-react'

import { submitFeedback } from '../api/api'
import type { AnomalyLabel, AnomalySegment, FeedbackSubmitRequest } from '../types/types'
import { ANOMALY_LABELS } from '../types/types'

interface FeedbackPanelProps {
  segment: AnomalySegment | null
  onFeedbackSubmitted: (updatedSegment: AnomalySegment) => void
}

type FeedbackState = 'form' | 'detail'
type LabelMode = 'label_correct' | 'edit'

const INCORRECT_LABEL_OPTIONS: AnomalyLabel[] = ['Normal', 'Other']
const ANOMALY_CATEGORY_OPTIONS = ANOMALY_LABELS.filter(
  (label) => label !== 'Normal' && label !== 'Other',
)

function formatSubmittedAt(value: string | null): string {
  if (value === null) {
    return '--'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })}`
}

function getCorrectEditLabelOptions(predictedLabel: AnomalyLabel): AnomalyLabel[] {
  return [
    ...ANOMALY_CATEGORY_OPTIONS.filter((label) => label !== predictedLabel),
    'Other',
  ]
}

function getEditableLabelOptions(
  predictedLabel: AnomalyLabel,
  isCorrect: boolean | null,
): AnomalyLabel[] {
  if (isCorrect === false) {
    return INCORRECT_LABEL_OPTIONS
  }

  return getCorrectEditLabelOptions(predictedLabel)
}

function getDefaultEditedLabel(
  predictedLabel: AnomalyLabel,
  isCorrect: boolean | null,
): AnomalyLabel {
  return getEditableLabelOptions(predictedLabel, isCorrect)[0] ?? 'Other'
}

function createInitialFormState(segment: AnomalySegment) {
  const hasFeedback = segment.feedback_submitted_at !== null
  const initialIsCorrect =
    hasFeedback && segment.is_correct !== null ? segment.is_correct === 1 : null
  const savedVerifiedLabel = segment.verified_label ?? segment.predicted_class
  const labelMode: LabelMode | null = hasFeedback
    ? savedVerifiedLabel === segment.predicted_class && segment.is_correct !== 0
      ? 'label_correct'
      : 'edit'
    : null
  const editableOptions = getEditableLabelOptions(segment.predicted_class, initialIsCorrect)
  const verifiedLabel =
    labelMode === 'edit' && !editableOptions.includes(savedVerifiedLabel)
      ? getDefaultEditedLabel(segment.predicted_class, initialIsCorrect)
      : savedVerifiedLabel

  return {
    isCorrect: initialIsCorrect,
    labelMode,
    verifiedLabel,
    otherDescription: segment.other_description ?? '',
    investigatorComment: segment.investigator_comment ?? '',
  }
}

export function FeedbackPanel({ segment, onFeedbackSubmitted }: FeedbackPanelProps) {
  const [feedbackState, setFeedbackState] = useState<FeedbackState>('form')
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null)
  const [labelMode, setLabelMode] = useState<LabelMode | null>(null)
  const [verifiedLabel, setVerifiedLabel] = useState<AnomalyLabel>('Normal')
  const [otherDescription, setOtherDescription] = useState('')
  const [investigatorComment, setInvestigatorComment] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!segment) {
      setFeedbackState('form')
      setIsCorrect(null)
      setLabelMode(null)
      setVerifiedLabel('Normal')
      setOtherDescription('')
      setInvestigatorComment('')
      setError(null)
      return
    }

    const initialFormState = createInitialFormState(segment)
    setFeedbackState(segment.feedback_submitted_at ? 'detail' : 'form')
    setIsCorrect(initialFormState.isCorrect)
    setLabelMode(initialFormState.labelMode)
    setVerifiedLabel(initialFormState.verifiedLabel)
    setOtherDescription(initialFormState.otherDescription)
    setInvestigatorComment(initialFormState.investigatorComment)
    setError(null)
  }, [segment])

  if (!segment) {
    return (
      <section className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
        <h3 className="mb-4 border-b border-[#C3C6D7] pb-4 text-lg font-semibold text-[#131B2E]">
          Feedback & Validation
        </h3>
        <p className="text-sm text-[#434655]">Select a segment before submitting feedback.</p>
      </section>
    )
  }

  const activeSegment = segment
  const editableLabels = getEditableLabelOptions(activeSegment.predicted_class, isCorrect)
  const isEditing = activeSegment.feedback_submitted_at !== null && feedbackState === 'form'
  const selectedVerifiedLabel =
    labelMode === 'edit' ? verifiedLabel : activeSegment.predicted_class
  const isOtherMissing = selectedVerifiedLabel === 'Other' && otherDescription.trim().length === 0
  const hasRequiredSelections = isCorrect !== null && labelMode !== null
  const canSubmit = hasRequiredSelections && !isOtherMissing && !isSubmitting

  function resetFormToSegmentFeedback() {
    const initialFormState = createInitialFormState(activeSegment)
    setIsCorrect(initialFormState.isCorrect)
    setLabelMode(initialFormState.labelMode)
    setVerifiedLabel(initialFormState.verifiedLabel)
    setOtherDescription(initialFormState.otherDescription)
    setInvestigatorComment(initialFormState.investigatorComment)
    setError(null)
  }

  function handleCorrectnessChange(nextIsCorrect: boolean) {
    setIsCorrect(nextIsCorrect)
    if (!nextIsCorrect) {
      setLabelMode('edit')
      setVerifiedLabel('Normal')
      setOtherDescription('')
      return
    }

    if (labelMode === 'edit') {
      const nextOptions = getEditableLabelOptions(activeSegment.predicted_class, true)
      if (!nextOptions.includes(verifiedLabel)) {
        setVerifiedLabel(getDefaultEditedLabel(activeSegment.predicted_class, true))
      }
    }
  }

  function handleLabelModeChange(nextMode: LabelMode) {
    if (isCorrect === false && nextMode === 'label_correct') {
      return
    }

    setLabelMode(nextMode)
    if (nextMode === 'label_correct') {
      setVerifiedLabel(activeSegment.predicted_class)
      setOtherDescription('')
    } else {
      const nextOptions = getEditableLabelOptions(activeSegment.predicted_class, isCorrect)
      if (!nextOptions.includes(verifiedLabel)) {
        setVerifiedLabel(getDefaultEditedLabel(activeSegment.predicted_class, isCorrect))
      }
    }
  }

  async function handleSubmit() {
    if (!canSubmit) {
      return
    }

    if (isCorrect === null) {
      return
    }

    const payload: FeedbackSubmitRequest = {
      is_correct: isCorrect,
      verified_label: selectedVerifiedLabel,
      other_description:
        selectedVerifiedLabel === 'Other' ? otherDescription.trim() : null,
      investigator_comment: investigatorComment.trim() || null,
    }

    setIsSubmitting(true)
    setError(null)
    try {
      const response = await submitFeedback(activeSegment.id, payload)
      if (!response.success || response.data === null) {
        throw new Error(response.message)
      }

      onFeedbackSubmitted(response.data)
      setFeedbackState('detail')
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Feedback failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  if (feedbackState === 'detail' && activeSegment.feedback_submitted_at !== null) {
    return (
      <section className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
        <h3 className="mb-4 border-b border-[#C3C6D7] pb-4 text-lg font-semibold text-[#131B2E]">
          Feedback Detail
        </h3>
        <div className="flex flex-col gap-4 text-sm">
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Segment Detect</span>
            <span className="inline-flex items-center gap-2 font-semibold text-[#131B2E]">
              {activeSegment.is_correct === 1 ? (
                <Check className="h-4 w-4 text-emerald-800" aria-hidden="true" />
              ) : (
                <X className="h-4 w-4 text-red-800" aria-hidden="true" />
              )}
              {activeSegment.is_correct === 1 ? 'Correct' : 'Incorrect'}
            </span>
          </div>
          <div className="flex items-start justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Verified Label</span>
            <span className="text-right font-semibold text-[#131B2E]">
              {activeSegment.verified_label ?? '--'}
              {activeSegment.verified_label === 'Other' && activeSegment.other_description ? (
                <span className="mt-1 block text-xs font-medium text-[#434655]">
                  {activeSegment.other_description}
                </span>
              ) : null}
            </span>
          </div>
          <div className="flex items-start justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Investigator Comments</span>
            <span className="text-right font-semibold text-[#131B2E]">
              {activeSegment.investigator_comment ?? '--'}
            </span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs font-medium text-[#737686]">Submitted At</span>
            <span className="font-semibold text-[#131B2E]">
              {formatSubmittedAt(activeSegment.feedback_submitted_at)}
            </span>
          </div>
        </div>
        <button
          type="button"
          className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100"
          onClick={() => {
            resetFormToSegmentFeedback()
            setFeedbackState('form')
          }}
        >
          <Pencil className="h-4 w-4" aria-hidden="true" />
          Edit Feedback
        </button>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-[#C3C6D7] bg-white p-6 shadow-sm">
      <h3 className="mb-4 border-b border-[#C3C6D7] pb-4 text-lg font-semibold text-[#131B2E]">
        Feedback & Validation
      </h3>

      <div className="flex flex-col gap-4">
        <div>
          <p className="mb-2 text-sm text-[#131B2E]">Is the detected anomaly segment correct?</p>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              className={
                isCorrect === true
                  ? 'inline-flex items-center gap-2 rounded-lg bg-[#004AC6] px-4 py-2 text-sm font-semibold text-white'
                  : 'inline-flex items-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100'
              }
              onClick={() => handleCorrectnessChange(true)}
            >
              <Check className="h-4 w-4" aria-hidden="true" />
              Correct
            </button>
            <button
              type="button"
              className={
                isCorrect === false
                  ? 'inline-flex items-center gap-2 rounded-lg bg-[#004AC6] px-4 py-2 text-sm font-semibold text-white'
                  : 'inline-flex items-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100'
              }
              onClick={() => handleCorrectnessChange(false)}
            >
              <X className="h-4 w-4" aria-hidden="true" />
              Incorrect
            </button>
          </div>
        </div>

        <div>
          <p className="mb-2 text-sm text-[#131B2E]">Is the predicted activity correct?</p>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              className={
                labelMode === 'label_correct'
                  ? 'inline-flex items-center gap-2 rounded-lg bg-[#004AC6] px-4 py-2 text-sm font-semibold text-white'
                  : 'inline-flex items-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50'
              }
              onClick={() => handleLabelModeChange('label_correct')}
              disabled={isCorrect === false}
            >
              <Check className="h-4 w-4" aria-hidden="true" />
              Label Correct
            </button>
            <button
              type="button"
              className={
                labelMode === 'edit'
                  ? 'inline-flex items-center gap-2 rounded-lg bg-[#004AC6] px-4 py-2 text-sm font-semibold text-white'
                  : 'inline-flex items-center gap-2 rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100'
              }
              onClick={() => handleLabelModeChange('edit')}
            >
              <Pencil className="h-4 w-4" aria-hidden="true" />
              Edit Label
            </button>
          </div>

          {labelMode === 'edit' ? (
            <div className="mt-2 flex flex-col gap-2">
              <select
                className="w-full rounded-lg border border-[#C3C6D7] bg-white px-3 py-2 text-sm text-[#131B2E] outline-none focus:ring-2 focus:ring-current"
                value={verifiedLabel}
                onChange={(event) => {
                  const nextLabel = event.target.value as AnomalyLabel
                  setVerifiedLabel(nextLabel)
                  if (nextLabel !== 'Other') {
                    setOtherDescription('')
                  }
                }}
              >
                {editableLabels.map((label) => (
                  <option key={label} value={label}>
                    {label}
                  </option>
                ))}
              </select>
              {verifiedLabel === 'Other' ? (
                <textarea
                  className="min-h-[96px] w-full resize-none rounded-lg border border-[#C3C6D7] bg-white px-3 py-2 text-sm text-[#131B2E] outline-none focus:ring-2 focus:ring-current"
                  placeholder="Describe the activity"
                  value={otherDescription}
                  onChange={(event) => setOtherDescription(event.target.value)}
                  required
                />
              ) : null}
            </div>
          ) : null}
        </div>

        <div>
          <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-[#737686]">
            Investigator Comments
          </label>
          <textarea
            className="min-h-[96px] w-full resize-none rounded-lg border border-[#C3C6D7] bg-white px-3 py-2 text-sm text-[#131B2E] outline-none focus:ring-2 focus:ring-current"
            placeholder="Describe findings, involved parties..."
            value={investigatorComment}
            onChange={(event) => setInvestigatorComment(event.target.value)}
          />
        </div>

        {isOtherMissing ? (
          <p className="text-sm font-medium text-red-800">Other description is required.</p>
        ) : null}
        {!hasRequiredSelections ? (
          <p className="text-sm font-medium text-[#434655]">Please answer both questions.</p>
        ) : null}
        {error ? <p className="text-sm font-medium text-red-800">{error}</p> : null}

        <div className="flex flex-col gap-2">
          <button
            type="button"
            className="inline-flex w-full items-center justify-center rounded-lg bg-[#004AC6] px-4 py-3 text-base font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void handleSubmit()}
            disabled={!canSubmit}
          >
            {isSubmitting ? 'Saving...' : isEditing ? 'Save Changes' : 'Submit Feedback'}
          </button>
          {isEditing ? (
            <button
              type="button"
              className="inline-flex w-full items-center justify-center rounded-lg border border-[#C3C6D7] bg-white px-4 py-2 text-sm font-semibold text-[#434655] transition hover:bg-slate-100"
              onClick={() => {
                resetFormToSegmentFeedback()
                setFeedbackState('detail')
              }}
            >
              Cancel
            </button>
          ) : null}
        </div>
      </div>
    </section>
  )
}
