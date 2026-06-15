import type { ReactNode } from 'react'
import { ChevronDown, Download, Filter } from 'lucide-react'

import { getUploadUrl } from '../../api/api'
import type { InvestigationItem, InvestigationStatus } from '../../types/types'

interface RecentInvestigationsProps {
  items: InvestigationItem[]
  totalCount: number
  filterPanel?: ReactNode
  onLoadMore: () => void
  onRowClick: (videoId: string) => void
  onFilterClick: () => void
  onExportData: () => void
}

type ConfidenceSeverity = 'HIGH' | 'MEDIUM' | 'LOW'

const CONFIDENCE_FILL: Record<ConfidenceSeverity, string> = {
  HIGH: '#BA1A1A',
  MEDIUM: '#F59E0B',
  LOW: '#004AC6',
}

const CONFIDENCE_TEXT: Record<ConfidenceSeverity, string> = {
  HIGH: '#BA1A1A',
  MEDIUM: '#D97706',
  LOW: '#004AC6',
}

/** Updated status badge colors — VALIDATED uses green per Figma */
const STATUS_BADGE: Record<InvestigationStatus, { bg: string; text: string }> = {
  'HIGH ALERT': { bg: 'rgba(186, 26, 26, 0.10)', text: '#BA1A1A' },
  'IN REVIEW': { bg: 'rgba(80, 95, 118, 0.10)', text: '#505F76' },
  VALIDATED: { bg: 'rgba(16, 185, 129, 0.10)', text: '#059669' },
}

function getConfidenceSeverity(confidence: number): ConfidenceSeverity {
  if (confidence >= 0.85) {
    return 'HIGH'
  }
  if (confidence >= 0.65) {
    return 'MEDIUM'
  }
  return 'LOW'
}

function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(0)}%`
}

function formatDuration(value: number | null): string {
  if (value === null) {
    return '0:00 min'
  }

  const rounded = Math.max(0, Math.round(value))
  const minutes = Math.floor(rounded / 60)
  const seconds = rounded % 60
  return `${minutes}:${String(seconds).padStart(2, '0')} min`
}

function formatCreatedTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  const today = new Date()
  const isToday =
    date.getFullYear() === today.getFullYear() &&
    date.getMonth() === today.getMonth() &&
    date.getDate() === today.getDate()

  if (isToday) {
    return `Today, ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
  }

  return date.toLocaleDateString('en-GB')
}

export function RecentInvestigations({
  items,
  totalCount,
  filterPanel,
  onLoadMore,
  onRowClick,
  onFilterClick,
  onExportData,
}: RecentInvestigationsProps) {
  const canLoadMore = items.length < totalCount

  return (
    <article
      className="overflow-hidden rounded-xl bg-white"
      style={{
        boxShadow: '0px 1px 2px rgba(0, 0, 0, 0.05)',
        outline: '1px #C3C6D7 solid',
        outlineOffset: '-1px',
        borderRadius: 12,
      }}
    >
      {/* Header — 32px padding with border-bottom per Figma */}
      <div
        className="flex flex-col gap-4 bg-white lg:flex-row lg:items-center lg:justify-between"
        style={{
          padding: 32,
          borderBottom: '1px #C3C6D7 solid',
        }}
      >
        <div>
          <h2 className="dashboard-section-title text-[#131B2E]">Recent Investigations</h2>
          <p
            style={{ fontSize: 14, fontWeight: 400, lineHeight: '20px', color: '#434655' }}
          >
            Detailed log of AI-assisted surveillance analysis
          </p>
        </div>
        <div className="flex flex-wrap items-center" style={{ gap: 16 }}>
          <button
            type="button"
            className="inline-flex items-center justify-center rounded-lg bg-white hover:bg-gray-50"
            style={{
              paddingLeft: 16,
              paddingRight: 16,
              paddingTop: 8,
              paddingBottom: 8,
              outline: '1px #C3C6D7 solid',
              outlineOffset: '-1px',
              gap: 8,
              fontSize: 14,
              fontWeight: 400,
              lineHeight: '20px',
              color: '#131B2E',
            }}
            onClick={onFilterClick}
          >
            <Filter className="h-4 w-4" aria-hidden="true" />
            Filter
          </button>
          <button
            type="button"
            className="inline-flex items-center justify-center rounded-lg bg-white hover:bg-gray-50"
            style={{
              paddingLeft: 16,
              paddingRight: 16,
              paddingTop: 8,
              paddingBottom: 8,
              outline: '1px #C3C6D7 solid',
              outlineOffset: '-1px',
              gap: 8,
              fontSize: 14,
              fontWeight: 400,
              lineHeight: '20px',
              color: '#131B2E',
            }}
            onClick={(event) => {
              event.stopPropagation()
              onExportData()
            }}
          >
            <Download className="h-4 w-4" aria-hidden="true" />
            Export Data
          </button>
        </div>
      </div>

      {filterPanel}

      <div className="overflow-x-auto">
        <table className="w-full border-collapse" style={{ minWidth: 980 }}>
          <thead>
            <tr style={{ background: 'rgba(242, 243, 255, 0.30)' }}>
              <th
                className="text-left"
                style={{
                  padding: '24px 24px 24px 32px',
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                  width: 260,
                }}
              >
                Video Name
              </th>
              <th
                className="text-left"
                style={{
                  padding: '24px 16px',
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                  width: 160,
                }}
              >
                Detected
                <br />
                Activity
              </th>
              <th
                className="text-left"
                style={{
                  padding: '24px 16px',
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                  width: 173,
                }}
              >
                Confidence
              </th>
              <th
                className="text-left"
                style={{
                  padding: '24px 16px',
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                  width: 140,
                }}
              >
                Review
                <br />
                Status
              </th>
              <th
                className="text-left"
                style={{
                  padding: '24px 16px',
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                  width: 114,
                }}
              >
                Created
                <br />
                Time
              </th>
              <th
                className="text-left"
                style={{
                  padding: '24px 32px 24px 16px',
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                  width: 170,
                }}
              >
                Action
              </th>
            </tr>
          </thead>
          <tbody>
            {items.length > 0 ? (
              items.map((item) => {
                const severity = getConfidenceSeverity(item.confidence)
                const statusStyle = STATUS_BADGE[item.investigation_status]
                const widthPct = Math.min(100, Math.max(0, item.confidence * 100))
                const previewUrl = item.file_path ? `${getUploadUrl(item.file_path)}#t=0.1` : ''
                return (
                  <tr
                    key={item.video_id}
                    className="cursor-pointer hover:bg-gray-50"
                    style={{ borderTop: '1px #C3C6D7 solid' }}
                    onClick={() => onRowClick(item.video_id)}
                  >
                    {/* Video Name */}
                    <td style={{ paddingLeft: 32, paddingRight: 24 }}>
                      <div className="flex items-center" style={{ gap: 16 }}>
                        <div
                          className="flex shrink-0 items-center justify-center overflow-hidden"
                          style={{
                            width: 48,
                            height: 48,
                            background: '#E2E8F0',
                            borderRadius: 4,
                          }}
                        >
                          {previewUrl ? (
                            <video
                              className="h-full w-full object-cover"
                              src={previewUrl}
                              preload="metadata"
                              muted
                              playsInline
                              aria-label={`${item.filename} preview`}
                            />
                          ) : null}
                        </div>
                        <div className="min-w-0">
                          <p
                            className="truncate"
                            style={{
                              maxWidth: 160,
                              fontSize: 16,
                              fontWeight: 600,
                              lineHeight: '20px',
                              color: '#131B2E',
                            }}
                          >
                            {item.filename}
                          </p>
                          <p
                            style={{
                              marginTop: 1,
                              fontSize: 12,
                              fontWeight: 400,
                              lineHeight: '14px',
                              color: '#737686',
                            }}
                          >
                            {formatDuration(item.duration)}
                          </p>
                        </div>
                      </div>
                    </td>
                    {/* Detected Activity */}
                    <td
                      style={{
                        paddingLeft: 16,
                        paddingRight: 16,
                        paddingTop: 40,
                        paddingBottom: 40,
                        fontSize: 16,
                        fontWeight: 400,
                        color: '#131B2E',
                      }}
                    >
                      {item.detected_activity}
                    </td>
                    {/* Confidence — bar + percentage */}
                    <td style={{ paddingLeft: 16, paddingRight: 16 }}>
                      <div className="flex items-center" style={{ gap: 8 }}>
                        <div
                          className="relative overflow-hidden rounded-full"
                          style={{ width: 64, height: 6, background: '#E2E7FF' }}
                        >
                          <div
                            className="absolute left-0 top-0"
                            style={{
                              height: 6,
                              width: `${widthPct}%`,
                              background: CONFIDENCE_FILL[severity],
                            }}
                          />
                        </div>
                        <span
                          style={{
                            fontSize: 16,
                            fontWeight: 700,
                            color: CONFIDENCE_TEXT[severity],
                          }}
                        >
                          {formatConfidence(item.confidence)}
                        </span>
                      </div>
                    </td>
                    {/* Review Status badge */}
                    <td style={{ paddingLeft: 16, paddingRight: 16 }}>
                      <span
                        className="inline-flex items-center rounded-full"
                        style={{
                          paddingLeft: 10,
                          paddingRight: 10,
                          paddingTop: 5,
                          paddingBottom: 5,
                          fontSize: 10,
                          fontWeight: 700,
                          lineHeight: '12px',
                          textTransform: 'uppercase',
                          letterSpacing: 0.5,
                          background: statusStyle.bg,
                          color: statusStyle.text,
                        }}
                      >
                        {item.investigation_status}
                      </span>
                    </td>
                    {/* Created Time */}
                    <td
                      style={{
                        padding: '32px 16px',
                        fontSize: 16,
                        fontWeight: 400,
                        color: '#434655',
                      }}
                    >
                      {formatCreatedTime(item.created_at)}
                    </td>
                    {/* Action */}
                    <td style={{ padding: '32px 32px 32px 16px' }} className="text-left">
                      <button
                        type="button"
                        className="inline-flex cursor-pointer items-center justify-center bg-white p-0 hover:opacity-80"
                        style={{
                          border: 0,
                          fontSize: 16,
                          fontWeight: 600,
                          lineHeight: '24px',
                          color: '#004AC6',
                        }}
                        onClick={(event) => {
                          event.stopPropagation()
                          onRowClick(item.video_id)
                        }}
                      >
                        View Investigation
                      </button>
                    </td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td
                  className="text-center text-sm"
                  colSpan={6}
                  style={{ padding: '48px 32px', color: '#505F76' }}
                >
                  No investigations found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {canLoadMore ? (
        <div
          className="flex items-start justify-center"
          style={{
            padding: 16,
            background: 'rgba(242, 243, 255, 0.20)',
            borderTop: '1px #C3C6D7 solid',
          }}
        >
          <button
            type="button"
            className="inline-flex cursor-pointer items-center bg-white p-0 hover:opacity-80"
            style={{
              border: 0,
              gap: 8,
              fontSize: 16,
              fontWeight: 600,
              lineHeight: '24px',
              color: '#004AC6',
            }}
            onClick={onLoadMore}
          >
            Load more investigations
            <ChevronDown style={{ width: 10, height: 10, color: '#004AC6' }} aria-hidden="true" />
          </button>
        </div>
      ) : null}
    </article>
  )
}
