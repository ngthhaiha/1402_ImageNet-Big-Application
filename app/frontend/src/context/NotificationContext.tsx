import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  getUnreadCount,
  getNotifications,
  patchNotificationRead,
  patchNotificationReadAll,
} from '../api/api'
import type { Notification } from '../types/types'

interface NotificationContextValue {
  notifications: Notification[]
  unreadCount: number
  markRead: (id: number) => void
  markAllRead: () => void
  loadMore: () => void
  dismissStack: () => void
  hasMore: boolean
  isLoading: boolean
  isStackDismissed: boolean
}

const NotificationContext = createContext<NotificationContextValue | null>(null)

interface NotificationProviderProps {
  children: ReactNode
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [unreadCount, setUnreadCount] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [limit, setLimit] = useState(5)
  const [isLoading, setIsLoading] = useState(false)
  const [dismissedStackKey, setDismissedStackKey] = useState('')

  const fetchNotifications = useCallback(async () => {
    setIsLoading(true)
    try {
      const [response, unread] = await Promise.all([
        getNotifications({ limit, offset: 0 }),
        getUnreadCount(),
      ])
      setNotifications(response.items)
      setTotalCount(response.total)
      setUnreadCount(unread)
    } finally {
      setIsLoading(false)
    }
  }, [limit])

  useEffect(() => {
    void fetchNotifications()

    const intervalId = window.setInterval(() => {
      void fetchNotifications()
    }, 10000)

    return () => window.clearInterval(intervalId)
  }, [fetchNotifications])

  const markRead = useCallback(
    (id: number) => {
      const target = notifications.find((notification) => notification.id === id)
      setNotifications((current) =>
        current.map((notification) =>
          notification.id === id
            ? { ...notification, is_read: true }
            : notification,
        ),
      )
      if (target && !target.is_read) {
        setUnreadCount((current) => Math.max(0, current - 1))
      }

      void patchNotificationRead(id).catch(() => {
        void fetchNotifications()
      })
    },
    [fetchNotifications, notifications],
  )

  const markAllRead = useCallback(() => {
    setNotifications((current) =>
      current.map((notification) => ({ ...notification, is_read: true })),
    )
    setUnreadCount(0)

    void patchNotificationReadAll().catch(() => {
      void fetchNotifications()
    })
  }, [fetchNotifications])

  const loadMore = useCallback(() => {
    setLimit((current) => current + 10)
  }, [])

  const anomalyUnreadKey = useMemo(
    () =>
      notifications
        .filter((notification) => !notification.is_read && notification.video_id !== null)
        .map((notification) => notification.id)
        .join(','),
    [notifications],
  )

  const dismissStack = useCallback(() => {
    setDismissedStackKey(anomalyUnreadKey)
  }, [anomalyUnreadKey])

  const hasMore = notifications.length < totalCount
  const isStackDismissed = anomalyUnreadKey !== '' && dismissedStackKey === anomalyUnreadKey

  const value = useMemo(
    () => ({
      notifications,
      unreadCount,
      markRead,
      markAllRead,
      loadMore,
      dismissStack,
      hasMore,
      isLoading,
      isStackDismissed,
    }),
    [
      notifications,
      unreadCount,
      markRead,
      markAllRead,
      loadMore,
      dismissStack,
      hasMore,
      isLoading,
      isStackDismissed,
    ],
  )

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

export function useNotifications() {
  const context = useContext(NotificationContext)

  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider')
  }

  return context
}
