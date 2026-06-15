import { useState } from 'react'

import { useNotifications } from '../../context/NotificationContext'
import { NotificationCard } from './NotificationCard'

const CARD_HEIGHT = 46

export function NotificationStack() {
  const { notifications, markRead, isStackDismissed } = useNotifications()
  const [isExpanded, setIsExpanded] = useState(false)

  const anomalyUnreadNotifications = notifications
    .filter((notification) => !notification.is_read && notification.video_id !== null)
  const unreadNotifications = anomalyUnreadNotifications.slice(0, 5)

  if (unreadNotifications.length === 0 || isStackDismissed) {
    return null
  }

  const collapsedNotifications = unreadNotifications.slice(0, 3)
  const collapsedHeight =
    CARD_HEIGHT + (Math.min(collapsedNotifications.length, 3) - 1) * 8

  return (
    <div
      className={`notification-stack ${
        isExpanded ? 'notification-stack-expanded' : 'notification-stack-collapsed'
      }`}
      style={isExpanded ? undefined : { height: `${collapsedHeight}px` }}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      {isExpanded
        ? unreadNotifications.map((notification) => (
            <NotificationCard
              key={notification.id}
              notification={notification}
              onRead={markRead}
            />
          ))
        : collapsedNotifications.map((notification, index) => (
            <div
              key={notification.id}
              style={{
                position: 'absolute',
                bottom: `${index * 8}px`,
                left: `${index * 4}px`,
                right: `${index * 4}px`,
                opacity: Math.max(0.7, 1 - index * 0.15),
                zIndex: 50 - index,
                transform: `scale(${1 - index * 0.02})`,
                transformOrigin: 'bottom center',
                transition: 'all 0.3s ease',
              }}
            >
              <NotificationCard notification={notification} onRead={markRead} />
            </div>
          ))}

      {!isExpanded && anomalyUnreadNotifications.length > 3 ? (
        <div className="notification-stack-more-badge">
          +{anomalyUnreadNotifications.length - 3} more
        </div>
      ) : null}
    </div>
  )
}
