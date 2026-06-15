import { useEffect, useRef, useState } from 'react'
import { Bell } from 'lucide-react'

import { useNotifications } from '../../context/NotificationContext'
import { NotificationDropdownItem } from './NotificationDropdownItem'

export function NotificationBell() {
  const {
    notifications,
    unreadCount,
    markRead,
    markAllRead,
    loadMore,
    dismissStack,
    hasMore,
    isLoading,
  } = useNotifications()
  const [isOpen, setIsOpen] = useState(false)
  const [isBadgeDismissed, setIsBadgeDismissed] = useState(false)
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const unreadKey = notifications
    .filter((notification) => !notification.is_read)
    .map((notification) => notification.id)
    .join(',')

  useEffect(() => {
    function handleMouseDown(event: MouseEvent) {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleMouseDown)
    return () => document.removeEventListener('mousedown', handleMouseDown)
  }, [])

  useEffect(() => {
    setIsBadgeDismissed(false)
  }, [unreadKey])

  function handleItemRead(id: number) {
    markRead(id)
    setIsOpen(false)
  }

  function handleBellClick() {
    dismissStack()
    setIsBadgeDismissed(true)
    setIsOpen((current) => !current)
  }

  return (
    <div className="notification-bell-wrapper" ref={wrapperRef}>
      <button
        type="button"
        className="page-header-icon-button notification-bell-button"
        onClick={handleBellClick}
        aria-label="Notifications"
      >
        <Bell className="page-header-icon" aria-hidden="true" />

        {unreadCount > 0 && !isBadgeDismissed ? (
          <div className="notification-bell-badge">
            {unreadCount > 99 ? '99+' : unreadCount}
          </div>
        ) : null}
      </button>

      {isOpen ? (
        <div className="notification-dropdown-panel">
          <div className="notification-dropdown-header">
            <h3>Notifications</h3>
            <button
              type="button"
              className="notification-text-action"
              onClick={markAllRead}
            >
              Mark all as read
            </button>
          </div>

          <div className="notification-dropdown-list">
            {isLoading && notifications.length === 0 ? (
              <div className="py-8 text-center text-sm text-[#737686]">
                Loading notifications...
              </div>
            ) : notifications.length === 0 ? (
              <div className="py-8 text-center text-sm text-[#737686]">
                No notifications
              </div>
            ) : (
              <>
                {notifications.map((notification) => (
                  <NotificationDropdownItem
                    key={notification.id}
                    notification={notification}
                    onRead={handleItemRead}
                  />
                ))}
                {hasMore ? (
                  <button
                    type="button"
                    className="notification-load-more"
                    onClick={loadMore}
                    disabled={isLoading}
                  >
                    {isLoading ? 'Loading...' : 'Load more notifications'}
                  </button>
                ) : null}
              </>
            )}
          </div>
        </div>
      ) : null}
    </div>
  )
}
