import type { AlertItem, ReviewStatus, Severity } from '../../types/types'

interface RecentAlertsProps {
  items: AlertItem[]
  onRowClick: (videoId: string) => void
  onViewAll: () => void
}

interface StatusDisplay {
  text: string
  badgeClass: string
}

const SEVERITY_CLASS: Record<Severity, string> = {
  HIGH: 'alert-severity-high',
  LOW: 'alert-severity-low',
  MEDIUM: 'alert-severity-medium',
}

function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function getStatusDisplay(
  reviewStatus: ReviewStatus | 'PROCESSING',
  isCorrect: boolean | null,
): StatusDisplay {
  if (reviewStatus === 'PENDING_REVIEW') {
    return {
      text: 'Unreviewed',
      badgeClass: 'alert-status-unreviewed',
    }
  }

  if (reviewStatus === 'LABEL_CORRECT') {
    return {
      text: 'Validated',
      badgeClass: 'alert-status-validated',
    }
  }

  if (reviewStatus === 'CORRECTED' && isCorrect === false) {
    return {
      text: 'False Positive',
      badgeClass: 'alert-status-muted',
    }
  }

  if (reviewStatus === 'CORRECTED') {
    return {
      text: 'Corrected',
      badgeClass: 'alert-status-primary',
    }
  }

  if (reviewStatus === 'LOGGED') {
    return {
      text: 'Logged',
      badgeClass: 'alert-status-primary',
    }
  }

  return {
    text: 'Processing',
    badgeClass: 'alert-status-muted',
  }
}

export function RecentAlerts({ items, onRowClick, onViewAll }: RecentAlertsProps) {
  const visibleItems = items.slice(0, 4)

  return (
    <article className="overflow-hidden rounded-xl border border-[rgba(195,198,215,0.30)] bg-white shadow-sm">
      {/* Header — 32px padding with border-bottom per Figma */}
      <div
        className="flex items-center justify-between bg-white"
        style={{
          padding: 32,
          borderBottom: '1px rgba(195, 198, 215, 0.30) solid',
        }}
      >
        <h2 className="dashboard-section-title text-[#131B2E]">Recent Alerts</h2>
        <button
          type="button"
          className="cursor-pointer bg-white p-0 hover:opacity-80"
          style={{
            border: 0,
            fontSize: 14,
            fontWeight: 500,
            letterSpacing: 0.6,
            lineHeight: '18px',
            color: '#004AC6',
          }}
          onClick={onViewAll}
        >
          View All
        </button>
      </div>

      <div className="overflow-x-auto" style={{ paddingBottom: 20 }}>
        <table className="w-full border-collapse" style={{ minWidth: 560 }}>
          <thead>
            <tr style={{ background: 'rgba(242, 243, 255, 0.50)' }}>
              <th
                className="text-left"
                style={{
                  width: 101,
                  padding: 16,
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                }}
              >
                Time
              </th>
              <th
                className="text-left"
                style={{
                  width: 149,
                  padding: 16,
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                }}
              >
                Activity Type
              </th>
              <th
                className="text-left"
                style={{
                  width: 108,
                  padding: 16,
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                }}
              >
                Confidence
              </th>
              <th
                className="text-left"
                style={{
                  width: 93,
                  padding: 16,
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                }}
              >
                Severity
              </th>
              <th
                className="text-left"
                style={{
                  width: 113,
                  padding: 16,
                  fontSize: 14,
                  fontWeight: 500,
                  letterSpacing: 0.6,
                  lineHeight: '18px',
                  color: '#737686',
                }}
              >
                Status
              </th>
            </tr>
          </thead>
          <tbody>
            {visibleItems.length > 0 ? (
              visibleItems.map((item) => {
                const status = getStatusDisplay(item.review_status, item.is_correct)
                return (
                  <tr
                    key={item.id}
                    className="cursor-pointer hover:bg-gray-50"
                    style={{ borderTop: '1px rgba(195, 198, 215, 0.30) solid' }}
                    onClick={() => onRowClick(item.video_id)}
                  >
                    <td style={{ padding: 16, fontSize: 16, fontWeight: 400, color: '#131B2E' }}>
                      {item.time}
                    </td>
                    <td style={{ padding: 16, fontSize: 16, fontWeight: 600, color: '#131B2E' }}>
                      {item.activity_type}
                    </td>
                    <td style={{ padding: 16, fontSize: 16, fontWeight: 400, color: '#131B2E' }}>
                      {formatConfidence(item.confidence)}
                    </td>
                    <td style={{ padding: 16 }}>
                      <span
                        className={`dashboard-alert-badge ${SEVERITY_CLASS[item.severity]}`}
                      >
                        {item.severity}
                      </span>
                    </td>
                    <td style={{ paddingLeft: 16 }}>
                      <span
                        className={`dashboard-alert-badge ${status.badgeClass}`}
                      >
                        {status.text}
                      </span>
                    </td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td
                  className="text-center text-sm"
                  colSpan={5}
                  style={{ padding: '48px 16px', color: '#505F76' }}
                >
                  No alerts found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </article>
  )
}
