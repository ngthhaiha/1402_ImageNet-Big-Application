import type { LucideIcon } from 'lucide-react'

interface StatsCardProps {
  icon: LucideIcon
  iconColor: string
  iconBg: string
  badge: string
  badgeColor: string
  badgeBg: string
  value: number
  valueColor: string
  label: string
  description: string
}

export function StatsCard({
  icon: Icon,
  iconColor,
  iconBg,
  badge,
  badgeColor,
  badgeBg,
  value,
  valueColor,
  label,
  description,
}: StatsCardProps) {
  return (
    <article className="dashboard-stats-card w-full rounded-xl border border-[rgba(195,198,215,0.30)] bg-white">
      <div className="flex items-start justify-between">
        <div className={`dashboard-stat-icon-box flex items-center justify-center rounded-lg ${iconBg}`}>
          <Icon
            className={`dashboard-stat-icon ${iconColor}`}
            aria-hidden="true"
            strokeWidth={2.25}
          />
        </div>

        <span
          className={`dashboard-stat-badge rounded px-2.5 py-1 font-bold ${badgeBg} ${badgeColor}`}
        >
          {badge}
        </span>
      </div>

      <p
        className={`dashboard-stat-value ${valueColor}`}
      >
        {value.toLocaleString()}
      </p>

      <p className="dashboard-stat-label whitespace-pre-line font-semibold text-[#434655]">
        {label}
      </p>

      <p className="dashboard-stat-description whitespace-pre-line font-medium tracking-[0.6px] text-[#737686]">
        {description}
      </p>
    </article>
  )
}
