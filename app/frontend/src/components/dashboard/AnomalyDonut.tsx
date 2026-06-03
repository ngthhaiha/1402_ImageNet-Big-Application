import { MoreHorizontal } from 'lucide-react'
import { Cell, Pie, PieChart, ResponsiveContainer } from 'recharts'

import { CLASS_COLORS, type DistributionItem } from '../../types/types'

interface AnomalyDonutProps {
  items: DistributionItem[]
}

function getClassColor(label: string): string {
  return CLASS_COLORS[label] ?? CLASS_COLORS.Other
}

export function AnomalyDonut({ items }: AnomalyDonutProps) {
  const sortedItems = [...items].sort((left, right) => {
    if (right.percentage !== left.percentage) {
      return right.percentage - left.percentage
    }
    if (right.count !== left.count) {
      return right.count - left.count
    }
    return left.class.localeCompare(right.class)
  })
  const total = sortedItems.reduce((sum, item) => sum + item.count, 0)

  return (
    <article className="rounded-xl border border-[rgba(195,198,215,0.30)] bg-white p-8 shadow-sm">
      <div className="mb-8 flex items-center justify-between">
        <h2 className="dashboard-section-title text-[#131B2E]">Anomaly Distribution</h2>
        <button
          type="button"
          className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-[#505F76] hover:bg-gray-50"
          aria-label="Distribution menu"
        >
          <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      <div className="flex items-center gap-8">
        {/* Donut chart — 192×192 per Figma */}
        <div className="relative shrink-0" style={{ width: 192, height: 192 }}>
          <ResponsiveContainer width="100%" height={192}>
            <PieChart>
              <Pie
                data={sortedItems}
                cx="50%"
                cy="50%"
                innerRadius={52}
                outerRadius={85}
                paddingAngle={2}
                dataKey="count"
                nameKey="class"
              >
                {sortedItems.map((entry) => (
                  <Cell key={entry.class} fill={getClassColor(entry.class)} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span style={{ fontSize: 24, fontWeight: 700, lineHeight: '32px', color: '#131B2E' }}>
              {total}
            </span>
            <span
              style={{
                fontSize: 12,
                fontWeight: 400,
                textTransform: 'uppercase',
                lineHeight: '16px',
                letterSpacing: 0.6,
                color: '#737686',
              }}
            >
              TOTAL
            </span>
          </div>
        </div>

        {/* Legend — vertical stack, each item has dot ring + name + percentage */}
        <div className="anomaly-donut-legend flex-1">
          {sortedItems.length > 0 ? (
            sortedItems.map((item) => (
              <div
                key={item.class}
                className="anomaly-donut-legend-item flex items-center"
              >
                {/* Dot with background ring */}
                <div
                  className="flex shrink-0 items-center justify-center rounded-full"
                  style={{
                    width: 12,
                    height: 12,
                    // No ring needed — just the dot per Figma legend
                  }}
                >
                  <div
                    className="rounded-full"
                    style={{
                      width: 12,
                      height: 12,
                      background: getClassColor(item.class),
                    }}
                  />
                </div>
                {/* Label + percentage stacked */}
                <div className="flex min-w-0 flex-col">
                  <span
                    className="truncate"
                    style={{
                      fontSize: 14,
                      fontWeight: 500,
                      lineHeight: '18px',
                      letterSpacing: 0.6,
                      color: '#131B2E',
                    }}
                  >
                    {item.class}
                  </span>
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 400,
                      lineHeight: '18px',
                      color: '#737686',
                    }}
                  >
                    {item.percentage.toFixed(0)}% ({item.count})
                  </span>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm" style={{ color: '#505F76' }}>No distribution data.</p>
          )}
        </div>
      </div>
    </article>
  )
}
