import type { LucideIcon } from 'lucide-react'

interface AlertStatsCardProps {
  label: string
  value: number
  subText: string
  subColor: string
  icon: LucideIcon
  trendIcon?: LucideIcon
  iconColor: string
}

export function AlertStatsCard({
  label,
  value,
  subText,
  subColor,
  icon: Icon,
  trendIcon: TrendIcon,
  iconColor,
}: AlertStatsCardProps) {
  return (
    <article className="alerts-card alerts-stat-card">
      <div className="alerts-stat-top">
        <p className="alerts-stat-label">{label}</p>
        <Icon className="alerts-stat-icon" style={{ color: iconColor }} aria-hidden="true" />
      </div>
      <div>
        <p className="alerts-stat-value">{value.toLocaleString()}</p>
        <div className="alerts-stat-sub" style={{ color: subColor }}>
          {TrendIcon ? <TrendIcon className="alerts-stat-trend-icon" aria-hidden="true" /> : null}
          <span>{subText}</span>
        </div>
      </div>
    </article>
  )
}
