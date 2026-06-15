import { AlertTriangle } from 'lucide-react'

import type { CriticalAlertItem, ReviewStatus } from '../../types/types'

interface CriticalAlertsTableProps {
  items: CriticalAlertItem[]
  onViewDetail: (videoId: string, segmentId: number) => void
}

function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

function isPending(status: ReviewStatus): boolean {
  return status === 'PENDING_REVIEW'
}

export function CriticalAlertsTable({
  items,
  onViewDetail,
}: CriticalAlertsTableProps) {
  const criticalItems = items.filter((item) => item.anomaly_score >= 0.85)

  return (
    <article className="alerts-critical-card">
      <div className="alerts-critical-header">
        <div className="alerts-critical-title">
          <AlertTriangle className="alerts-critical-title-icon" aria-hidden="true" />
          <h3>Recent Critical Alerts</h3>
        </div>
      </div>

      <div className="alerts-table-scroll">
        <table className="alerts-critical-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Activity</th>
              <th>Confidence</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {criticalItems.length > 0 ? (
              criticalItems.map((item) => {
                const pending = isPending(item.review_status)
                return (
                  <tr key={item.id}>
                    <td>{item.time}</td>
                    <td>
                      <div className="alerts-critical-activity">
                        <div>
                          <AlertTriangle className="alerts-critical-activity-icon" aria-hidden="true" />
                        </div>
                        <span>{item.activity_type}</span>
                      </div>
                    </td>
                    <td className="alerts-critical-confidence">
                      {formatConfidence(item.confidence_score)}
                    </td>
                    <td>
                      {pending ? (
                        <span className="alerts-critical-active">
                          <span aria-hidden="true" />
                          ACTIVE
                        </span>
                      ) : (
                        <span className="alerts-critical-archived">ARCHIVED</span>
                      )}
                    </td>
                    <td>
                      <button
                        type="button"
                        className={
                          pending
                            ? 'alerts-critical-button alerts-critical-button-primary'
                            : 'alerts-critical-button alerts-critical-button-secondary'
                        }
                        onClick={() => onViewDetail(item.video_id, item.id)}
                      >
                        View Detail
                      </button>
                    </td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td colSpan={5} className="alerts-empty-cell">
                  No critical alerts found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </article>
  )
}
