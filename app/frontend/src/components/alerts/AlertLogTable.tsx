import { Download, SlidersHorizontal } from 'lucide-react'

import type { AlertLogResponse, Severity } from '../../types/types'
import { getReviewStatusDisplay } from '../../utils/reviewStatus'

interface AlertLogTableProps {
  data: AlertLogResponse | null
  isLoading: boolean
  page: number
  onPageChange: (page: number) => void
  onViewInvestigation: (videoId: string, segmentId: number) => void
  onToggleFilter: () => void
  onExportCsv: () => void
  isFilterVisible: boolean
  isExporting: boolean
}

const SEVERITY_CLASS: Record<Severity, string> = {
  HIGH: 'alerts-severity-high',
  MEDIUM: 'alerts-severity-medium',
  LOW: 'alerts-severity-low',
}

function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

function getPageNumbers(currentPage: number, totalPages: number): number[] {
  if (totalPages <= 5) {
    return Array.from({ length: totalPages }, (_, index) => index + 1)
  }

  const start = Math.max(1, Math.min(currentPage - 2, totalPages - 4))
  return Array.from({ length: 5 }, (_, index) => start + index)
}

export function AlertLogTable({
  data,
  isLoading,
  page,
  onPageChange,
  onViewInvestigation,
  onToggleFilter,
  onExportCsv,
  isFilterVisible,
  isExporting,
}: AlertLogTableProps) {
  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = data?.total_pages ?? 0
  const pageNumbers = getPageNumbers(page, totalPages)

  return (
    <article className="alerts-table-card">
      <div className="alerts-table-title-row">
        <h3>Alert Log</h3>
        <div className="alerts-table-actions">
          <button
            type="button"
            className="alerts-icon-button"
            aria-label={isFilterVisible ? 'Hide alert filters' : 'Show alert filters'}
            aria-pressed={isFilterVisible}
            onClick={onToggleFilter}
          >
            <SlidersHorizontal className="alerts-small-icon" aria-hidden="true" />
          </button>
          <button
            type="button"
            className="alerts-icon-button"
            aria-label="Export alert log CSV"
            onClick={onExportCsv}
            disabled={isExporting || isLoading || total === 0}
          >
            <Download className="alerts-small-icon" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="alerts-table-scroll">
        <table className="alerts-log-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Video Name</th>
              <th>Activity Type</th>
              <th>Confidence</th>
              <th>Severity</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {items.length > 0 ? (
              items.map((item) => {
                const status = getReviewStatusDisplay(
                  item.review_status,
                  item.is_correct,
                  item.verified_label,
                )
                return (
                  <tr key={item.id}>
                    <td>{item.time}</td>
                    <td>
                      <span className="alerts-video-name" title={item.filename}>
                        {item.filename}
                      </span>
                    </td>
                    <td>{item.activity_type}</td>
                    <td>{formatConfidence(item.confidence_score)}</td>
                    <td>
                      <span className={`alerts-severity-badge ${SEVERITY_CLASS[item.severity]}`}>
                        {item.severity}
                      </span>
                    </td>
                    <td>
                      <span className={`dashboard-alert-badge ${status.badgeClass}`}>
                        {status.label}
                      </span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="alerts-link-button"
                        onClick={() => onViewInvestigation(item.video_id, item.id)}
                      >
                        View Investigation
                      </button>
                    </td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td colSpan={7} className="alerts-empty-cell">
                  {isLoading ? 'Loading alert log...' : 'No alerts found.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="alerts-pagination-row">
        <span>Showing {items.length} of {total.toLocaleString()} results</span>
        <div className="alerts-pagination">
          <button
            type="button"
            className="alerts-page-button alerts-page-arrow"
            disabled={page <= 1}
            onClick={() => onPageChange(page - 1)}
            aria-label="Previous page"
          >
            {'<'}
          </button>
          {pageNumbers.map((pageNumber) => (
            <button
              key={pageNumber}
              type="button"
              className={
                pageNumber === page
                  ? 'alerts-page-button alerts-page-active'
                  : 'alerts-page-button'
              }
              onClick={() => onPageChange(pageNumber)}
            >
              {pageNumber}
            </button>
          ))}
          <button
            type="button"
            className="alerts-page-button alerts-page-arrow"
            disabled={totalPages === 0 || page >= totalPages}
            onClick={() => onPageChange(page + 1)}
            aria-label="Next page"
          >
            {'>'}
          </button>
        </div>
      </div>
    </article>
  )
}
