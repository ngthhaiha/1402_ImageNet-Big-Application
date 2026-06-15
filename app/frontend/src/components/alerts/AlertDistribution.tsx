import type { DistributionItem } from '../../types/types'

interface AlertDistributionProps {
  items: DistributionItem[]
}

function getDistributionLabel(item: DistributionItem): string {
  return item.predicted_class ?? item.class
}

export function AlertDistribution({ items }: AlertDistributionProps) {
  return (
    <article className="alerts-card alerts-distribution-card">
      <div className="alerts-card-header">
        <h3>Alert Distribution</h3>
      </div>

      <div className="alerts-distribution-list">
        {items.length > 0 ? (
          items.map((item) => (
            <div key={getDistributionLabel(item)} className="alerts-distribution-item">
              <div className="alerts-distribution-label-row">
                <span>{getDistributionLabel(item)}</span>
                <strong>{item.percentage}%</strong>
              </div>
              <div className="alerts-distribution-track">
                <div
                  className="alerts-distribution-fill"
                  style={{ width: `${Math.min(100, Math.max(0, item.percentage))}%` }}
                />
              </div>
            </div>
          ))
        ) : (
          <p className="alerts-empty-text">No distribution data.</p>
        )}
      </div>
    </article>
  )
}
