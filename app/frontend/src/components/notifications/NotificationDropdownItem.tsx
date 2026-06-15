import { useNavigate } from 'react-router-dom'

import type { Notification } from '../../types/types'
import { TYPE_COLORS } from '../../types/types'

interface NotificationDropdownItemProps {
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

export function NotificationDropdownItem({
  notification,
  onRead,
}: NotificationDropdownItemProps) {
  const navigate = useNavigate()

  function handleClick() {
    onRead(notification.id)
    if (notification.target_url) {
      navigate(notification.target_url)
    }
  }

  return (
    <div
      className={`notification-dropdown-item ${
        !notification.is_read
          ? 'notification-dropdown-item-unread'
          : 'notification-dropdown-item-read'
      }`}
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
      <div className="notification-dropdown-item-content">
        <p className="notification-dropdown-title">
          {notification.title}
        </p>
        <p className="notification-dropdown-message">
          {notification.message}
        </p>
        <p className="notification-dropdown-time">
          {getRelativeTime(notification.created_at)}
        </p>
      </div>
      <span
        className="notification-dropdown-dot"
        style={{ background: TYPE_COLORS[notification.type] }}
        aria-hidden="true"
      />
    </div>
  )
}
