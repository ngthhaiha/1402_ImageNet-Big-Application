import type { TopDetection } from '../../types/types'

interface TopDetectionsProps {
  items: TopDetection[]
}

export function TopDetections({ items }: TopDetectionsProps) {
  const maxCount = items.length > 0 ? Math.max(...items.map((d) => d.count)) : 1

  return (
    <article className="rounded-xl border border-[#C3C6D7] bg-white p-8 shadow-sm">
      <div className="mb-8">  
        <h2 className="dashboard-section-title text-[#131B2E]">Top Detections</h2>
      </div>
      {items.length > 0 ? (
        <div className="flex flex-col" style={{ gap: 16 }}>
          {items.map((item) => {
            const widthPct = maxCount > 0 ? (item.count / maxCount) * 100 : 0
            return (
              <div key={item.class} className="flex flex-col" style={{ gap: 4 }}>
                {/* Label + count row */}
                <div className="flex items-start justify-between">
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      lineHeight: '16px',
                      color: '#131B2E',
                    }}
                  >
                    {item.class}
                  </span>
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 400,
                      lineHeight: '16px',
                      color: '#737686',
                    }}
                  >
                    {item.count}
                  </span>
                </div>
                {/* Progress bar */}
                <div
                  className="relative overflow-hidden rounded-full"
                  style={{ height: 8, background: '#E2E7FF' }}
                >
                  <div
                    className="absolute left-0 top-0 rounded-full"
                    style={{
                      height: 8,
                      width: `${widthPct}%`,
                      background: '#004AC6',
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="flex items-center justify-center" style={{ height: 240 }}>
          <span className="text-sm" style={{ color: '#505F76' }}>No detections found.</span>
        </div>
      )}
    </article>
  )
}
