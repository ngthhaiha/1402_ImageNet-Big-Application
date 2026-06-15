import { useNavigate } from 'react-router-dom'

import type { Notification } from '../../types/types'

interface NotificationCardProps {
  notification: Notification
  onRead: (id: number) => void
}

function getRelativeTime(value: string): string {
  const hasExplicitTimezone = /(?:z|[+-]\d{2}:\d{2})$/i.test(value)
  const normalizedValue = hasExplicitTimezone ? value : `${value}Z`
  const createdAt = new Date(normalizedValue).getTime()
  if (Number.isNaN(createdAt)) {
    return 'Just now'
  }

  const diffMs = Date.now() - createdAt
  const diffMinutes = Math.floor(diffMs / 60_000)

  if (diffMinutes < 60) {
    return `${Math.max(1, diffMinutes)}m ago`
  }

  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) {
    return `${diffHours}h ago`
  }

  return `${Math.floor(diffHours / 24)}d ago`
}

export function NotificationCard({ notification, onRead }: NotificationCardProps) {
  const navigate = useNavigate()

  function handleClick() {
    onRead(notification.id)
    if (notification.target_url) {
      navigate(notification.target_url)
    }
  }

  return (
    <div
      className="notification-stack-card"
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          handleClick()
        }
      }}
    >
      <p className="notification-stack-message">
        <span>{notification.message}</span>
        <span className="notification-stack-time">
          {getRelativeTime(notification.created_at)}
        </span>
      </p>
    </div>
  )
}
