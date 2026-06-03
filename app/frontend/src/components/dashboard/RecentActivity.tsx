import type { ActivityItem, DashboardActivityType } from '../../types/types'

interface RecentActivityProps {
  items: ActivityItem[]
}

/** Dot + ring colors per activity type — matching Figma exactly */
const DOT_STYLE_BY_TYPE: Record<
  DashboardActivityType,
  { dot: string; ring: string }
> = {
  UPLOAD: { dot: '#004AC6', ring: 'rgba(0, 74, 198, 0.10)' },
  REVIEW_COMPLETE: { dot: '#10B981', ring: 'rgba(16, 185, 129, 0.10)' },
  FLAG: { dot: '#943700', ring: 'rgba(148, 55, 0, 0.10)' },
}

/** Fallback for unknown types (e.g. EXPORT hard-coded in Figma) */
const DEFAULT_DOT_STYLE = { dot: '#505F76', ring: 'rgba(80, 95, 118, 0.10)' }

function getDotStyle(type: DashboardActivityType) {
  return DOT_STYLE_BY_TYPE[type] ?? DEFAULT_DOT_STYLE
}

function formatRelativeTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  const diffSeconds = Math.max(0, Math.floor((Date.now() - date.getTime()) / 1000))
  const diffMinutes = Math.floor(diffSeconds / 60)
  if (diffMinutes < 60) {
    return `${Math.max(1, diffMinutes)}m ago`
  }

  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) {
    return `${diffHours}h ago`
  }

  if (diffHours < 48) {
    return 'Yesterday'
  }

  return date.toLocaleDateString('en-GB')
}

export function RecentActivity({ items }: RecentActivityProps) {
  return (
    <article className="rounded-xl border border-[rgba(195,198,215,0.30)] bg-white p-8 shadow-sm">
      <div className="mb-8">
        <h2 className="dashboard-section-title text-[#131B2E]">Recent Activity</h2>
      </div>
      <div className="flex flex-col" style={{ gap: 20 }}>
        {items.length > 0 ? (
          items.map((item, index) => {
            const dotStyle = getDotStyle(item.type)
            return (
              <div
                key={`${item.type}-${item.created_at}-${index}`}
                className="flex items-start"
                style={{ gap: 16 }}
              >
                {/* Dot with ring container — 24×24 per Figma */}
                <div
                  className="flex shrink-0 items-center justify-center rounded-full"
                  style={{
                    width: 24,
                    height: 24,
                    background: dotStyle.ring,
                  }}
                >
                  <div
                    className="rounded-full"
                    style={{
                      width: 8,
                      height: 8,
                      background: dotStyle.dot,
                    }}
                  />
                </div>
                <div className="flex flex-col">
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      lineHeight: '18px',
                      letterSpacing: 0.6,
                      color: '#131B2E',
                    }}
                  >
                    {item.title}
                  </span>
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 400,
                      lineHeight: '18px',
                      color: '#737686',
                    }}
                  >
                    {item.detail} {'\u2022'} {formatRelativeTime(item.created_at)}
                  </span>
                </div>
              </div>
            )
          })
        ) : (
          <p className="text-sm" style={{ color: '#505F76' }}>No activity yet.</p>
        )}
      </div>
    </article>
  )
}
