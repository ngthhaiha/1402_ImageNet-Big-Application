import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, ReactNode } from 'react'
import {
  AlertTriangle,
  Bell,
  Camera,
  CheckCircle2,
  CheckSquare,
  ChevronRight,
  ClipboardCheck,
  Clock,
  Globe,
  LogOut,
  Mail,
  MessageSquare,
  Pencil,
  Share2,
  Shield,
  UploadCloud,
  Users,
  Video,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { getProfileActivity, getProfileStats } from '../api/api'
import { PageHeader } from '../components/PageHeader'
import { Toast } from '../components/Toast'
import { useAuth } from '../context/AuthContext'
import type {
  DashboardActivityType,
  ProfileActivityItem,
  ProfileStats,
} from '../types/types'

const NOTIFICATION_STORAGE_KEY = 'notification_prefs'
const ACTIVITY_PAGE_SIZE = 15
const AVATAR_SIZE = 256
const AVATAR_QUALITY = 0.86

interface NotificationPrefs {
  critical_alerts: boolean
  case_updates: boolean
  login_history: boolean
}

const DEFAULT_NOTIFICATION_PREFS: NotificationPrefs = {
  critical_alerts: true,
  case_updates: false,
  login_history: true,
}

function assertData<T>(response: { success: boolean; data: T | null; message: string }): T {
  if (!response.success || response.data === null) {
    throw new Error(response.message)
  }

  return response.data
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unable to load profile data'
}

function loadNotificationPrefs(): NotificationPrefs {
  try {
    const saved = window.localStorage.getItem(NOTIFICATION_STORAGE_KEY)
    if (saved === null) {
      return DEFAULT_NOTIFICATION_PREFS
    }

    const parsed = JSON.parse(saved) as Partial<NotificationPrefs>
    return {
      critical_alerts: parsed.critical_alerts ?? DEFAULT_NOTIFICATION_PREFS.critical_alerts,
      case_updates: parsed.case_updates ?? DEFAULT_NOTIFICATION_PREFS.case_updates,
      login_history: parsed.login_history ?? DEFAULT_NOTIFICATION_PREFS.login_history,
    }
  } catch {
    return DEFAULT_NOTIFICATION_PREFS
  }
}

function saveNotificationPrefs(nextPrefs: NotificationPrefs) {
  window.localStorage.setItem(NOTIFICATION_STORAGE_KEY, JSON.stringify(nextPrefs))
}

function formatRelativeTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  const diffMs = Math.max(0, Date.now() - date.getTime())
  if (diffMs < 3_600_000) {
    return `${Math.max(1, Math.floor(diffMs / 60_000))}m ago`
  }
  if (diffMs < 86_400_000) {
    return `${Math.floor(diffMs / 3_600_000)}h ago`
  }
  if (diffMs < 172_800_000) {
    return 'Yesterday'
  }

  return date.toLocaleDateString('en-GB')
}

function getActivityVideoName(description: string | null): string {
  if (!description) {
    return ''
  }

  const marker = ' in '
  const markerIndex = description.lastIndexOf(marker)
  return markerIndex === -1 ? description : description.slice(markerIndex + marker.length)
}

function formatActivityDescription(description: string | null): string {
  const videoName = getActivityVideoName(description)
  return videoName ? `Anomaly detected in ${videoName}` : 'Anomaly detected'
}

function getUserInitials(username: string | undefined): string {
  if (!username) {
    return 'U'
  }

  const parts = username
    .replace(/[_.-]+/g, ' ')
    .split(' ')
    .filter(Boolean)

  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase()
  }

  return username.slice(0, 2).toUpperCase()
}

function getAvatarStorageKey(userId: number): string {
  return `profile_avatar_${userId}`
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error('Could not read image file'))
      }
    }
    reader.onerror = () => reject(new Error('Could not read image file'))
    reader.readAsDataURL(file)
  })
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Could not load image file'))
    image.src = src
  })
}

async function createAvatarDataUrl(file: File): Promise<string> {
  const sourceDataUrl = await readFileAsDataUrl(file)
  const image = await loadImage(sourceDataUrl)
  const canvas = document.createElement('canvas')
  canvas.width = AVATAR_SIZE
  canvas.height = AVATAR_SIZE

  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('Could not prepare avatar image')
  }

  const sourceSize = Math.min(image.naturalWidth, image.naturalHeight)
  const sourceX = (image.naturalWidth - sourceSize) / 2
  const sourceY = (image.naturalHeight - sourceSize) / 2

  context.fillStyle = '#ffffff'
  context.fillRect(0, 0, AVATAR_SIZE, AVATAR_SIZE)
  context.drawImage(
    image,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    AVATAR_SIZE,
    AVATAR_SIZE,
  )

  return canvas.toDataURL('image/jpeg', AVATAR_QUALITY)
}

function getLanguageRegion(): string {
  const locale = navigator.language || 'en-US'
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'Local Time'
  const offsetMinutes = -new Date().getTimezoneOffset()
  const offsetSign = offsetMinutes >= 0 ? '+' : '-'
  const absoluteOffset = Math.abs(offsetMinutes)
  const offsetHours = Math.floor(absoluteOffset / 60)
  const offsetRemainder = absoluteOffset % 60
  const offsetLabel =
    offsetRemainder === 0
      ? `UTC${offsetSign}${offsetHours}`
      : `UTC${offsetSign}${offsetHours}:${String(offsetRemainder).padStart(2, '0')}`

  return `${locale} - ${timeZone} (${offsetLabel})`
}

function getActivityIcon(type: DashboardActivityType): LucideIcon {
  if (type === 'UPLOAD') {
    return UploadCloud
  }
  if (type === 'REVIEW_COMPLETE') {
    return ClipboardCheck
  }
  return AlertTriangle
}

function getActivityClass(type: DashboardActivityType): string {
  if (type === 'UPLOAD') {
    return 'profile-activity-icon-upload'
  }
  if (type === 'REVIEW_COMPLETE') {
    return 'profile-activity-icon-review'
  }
  return 'profile-activity-icon-flag'
}

function ProfileStatCard({
  icon: Icon,
  label,
  value,
  trend,
}: {
  icon: LucideIcon
  label: string
  value: number
  trend: string
}) {
  return (
    <article className="profile-stat-card">
      <div className="profile-stat-top">
        <span className="profile-stat-icon">
          <Icon aria-hidden="true" />
        </span>
        <span className="profile-stat-trend">{trend}</span>
      </div>
      <p className="profile-stat-label">{label}</p>
      <p className="profile-stat-value">{value}</p>
    </article>
  )
}

function ProfileCard({
  icon: Icon,
  title,
  children,
}: {
  icon: LucideIcon
  title: string
  children: ReactNode
}) {
  return (
    <article className="profile-card">
      <header className="profile-card-header">
        <Icon className="profile-card-title-icon" aria-hidden="true" />
        <h2>{title}</h2>
      </header>
      {children}
    </article>
  )
}

function SettingsRow({
  icon: Icon,
  label,
  value,
  badge,
  onClick,
}: {
  icon: LucideIcon
  label: string
  value?: string
  badge?: string
  onClick: () => void
}) {
  return (
    <button type="button" className="profile-settings-row" onClick={onClick}>
      <Icon className="profile-settings-icon" aria-hidden="true" />
      <span className="profile-settings-content">
        <span className="profile-settings-label">{label}</span>
        {value ? <span className="profile-settings-value">{value}</span> : null}
      </span>
      {badge ? <span className="profile-small-badge">{badge}</span> : null}
      <ChevronRight className="profile-chevron" aria-hidden="true" />
    </button>
  )
}

function Toggle({
  checked,
  onClick,
  label,
}: {
  checked: boolean
  onClick: () => void
  label: string
}) {
  return (
    <button
      type="button"
      className={`profile-toggle ${checked ? 'profile-toggle-on' : 'profile-toggle-off'}`}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={onClick}
    >
      <span className="profile-toggle-thumb" />
    </button>
  )
}

export function Profile() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const [stats, setStats] = useState<ProfileStats | null>(null)
  const [activity, setActivity] = useState<ProfileActivityItem[]>([])
  const [activityPage, setActivityPage] = useState(1)
  const [avatarSrc, setAvatarSrc] = useState<string | null>(null)
  const [notificationPrefs, setNotificationPrefs] = useState<NotificationPrefs>(
    loadNotificationPrefs,
  )
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toastOpen, setToastOpen] = useState(false)

  const statCards = useMemo(
    () => [
      {
        icon: Video,
        label: 'VIDEOS UPLOADED',
        value: stats?.videos_uploaded ?? 0,
        trend: '+12% vs LW',
      },
      {
        icon: CheckCircle2,
        label: 'CASES REVIEWED',
        value: stats?.cases_reviewed ?? 0,
        trend: '84% Completion',
      },
      {
        icon: MessageSquare,
        label: 'FEEDBACK SUBMITTED',
        value: stats?.feedback_submitted ?? 0,
        trend: '98% Avg Score',
      },
    ],
    [stats],
  )
  const displayName = user?.username ?? 'Loading user'
  const displayEmail = user?.email ?? 'Loading email'
  const displayUserId = user ? `ID: #USER-${user.id}` : 'ID: loading'
  const avatarLabel = `${displayName} avatar`
  const avatarInitials = getUserInitials(user?.username)
  const languageRegion = getLanguageRegion()
  const totalActivityPages = Math.max(1, Math.ceil(activity.length / ACTIVITY_PAGE_SIZE))
  const paginatedActivity = activity.slice(
    (activityPage - 1) * ACTIVITY_PAGE_SIZE,
    activityPage * ACTIVITY_PAGE_SIZE,
  )

  useEffect(() => {
    if (!user) {
      setAvatarSrc(null)
      return
    }

    setAvatarSrc(window.localStorage.getItem(getAvatarStorageKey(user.id)))
  }, [user])

  useEffect(() => {
    let isMounted = true

    async function loadProfileData() {
      setIsLoading(true)
      try {
        setError(null)
        const [statsResponse, activityResponse] = await Promise.all([
          getProfileStats(),
          getProfileActivity(50),
        ])

        if (!isMounted) {
          return
        }

        setStats(assertData(statsResponse))
        setActivity(assertData(activityResponse))
        setActivityPage(1)
      } catch (loadError) {
        if (isMounted) {
          setError(getErrorMessage(loadError))
        }
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

    void loadProfileData()

    return () => {
      isMounted = false
    }
  }, [])

  function showComingSoon() {
    setToastOpen(true)
  }

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  async function handleAvatarChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    if (!file.type.startsWith('image/')) {
      setToastOpen(true)
      event.target.value = ''
      return
    }

    try {
      const avatarDataUrl = await createAvatarDataUrl(file)
      setAvatarSrc(avatarDataUrl)
      if (user) {
        window.localStorage.setItem(getAvatarStorageKey(user.id), avatarDataUrl)
      }
    } catch {
      setToastOpen(true)
    }
    event.target.value = ''
  }

  function toggleNotification(key: keyof NotificationPrefs) {
    setNotificationPrefs((current) => {
      const nextPrefs = { ...current, [key]: !current[key] }
      saveNotificationPrefs(nextPrefs)
      return nextPrefs
    })
  }

  return (
    <section className="profile-page">
      <Toast
        open={toastOpen}
        title="Coming soon"
        message="This profile action is not available in the demo yet."
        variant="info"
        onClose={() => setToastOpen(false)}
      />

      <PageHeader pageName="Profile" />

      <div className="profile-shell">
        <section className="profile-header-card">
          <div className="profile-identity">
            <div className="profile-avatar-wrap">
              <div className="profile-avatar" aria-label={avatarLabel}>
                {avatarSrc ? (
                  <img src={avatarSrc} alt={avatarLabel} className="profile-avatar-image" />
                ) : (
                  avatarInitials
                )}
              </div>
              <label className="profile-camera-badge" aria-label="Upload avatar">
                <input
                  type="file"
                  accept="image/*"
                  className="profile-avatar-input"
                  onChange={handleAvatarChange}
                />
                <Camera aria-hidden="true" />
              </label>
            </div>

            <div className="profile-identity-text">
              <h1>{displayName}</h1>
              <p>{displayEmail}</p>
              <div className="profile-badges">
                <span className="profile-badge profile-badge-id">{displayUserId}</span>
                <span className="profile-badge profile-badge-location">Authenticated User</span>
                <span className="profile-badge profile-badge-shift">Active Session</span>
              </div>
            </div>
          </div>

          <div className="profile-header-actions">
            <button type="button" className="profile-primary-button" onClick={showComingSoon}>
              <Pencil aria-hidden="true" />
              Edit Profile
            </button>
            <button type="button" className="profile-outline-button" onClick={showComingSoon}>
              <Share2 aria-hidden="true" />
              Export Activity
            </button>
          </div>
        </section>

        {error ? <div className="profile-error">{error}</div> : null}

        <div className="profile-layout">
          <div className="profile-left-column">
            <div className="profile-stats-grid">
              {statCards.map((card) => (
                <ProfileStatCard key={card.label} {...card} />
              ))}
            </div>

            <article className="profile-activity-card">
              <header className="profile-card-header profile-activity-header">
                <UploadCloud className="profile-card-title-icon" aria-hidden="true" />
                <h2>Recent Activity</h2>
              </header>

              <div className="profile-activity-list">
                {activity.length > 0 ? (
                  paginatedActivity.map((item) => {
                    const Icon = getActivityIcon(item.type)
                    const isClickable = item.video_id !== null
                    return (
                      <div key={item.id} className="profile-activity-item">
                        <span className={`profile-activity-icon ${getActivityClass(item.type)}`}>
                          <Icon aria-hidden="true" />
                        </span>
                        <div className="profile-activity-body">
                          <div className="profile-activity-title-row">
                            {isClickable ? (
                              <button
                                type="button"
                                className="profile-activity-title-button"
                                onClick={() => navigate(`/videos/${item.video_id}`)}
                              >
                                {item.title}
                              </button>
                            ) : (
                              <span className="profile-activity-title">{item.title}</span>
                            )}
                            <span className="profile-activity-time">
                              {formatRelativeTime(item.created_at)}
                            </span>
                          </div>
                          <p className="profile-activity-description">
                            {formatActivityDescription(item.description)}
                          </p>
                          {item.type === 'REVIEW_COMPLETE' ? (
                            <span className="profile-critical-badge">CRITICAL</span>
                          ) : null}
                        </div>
                      </div>
                    )
                  })
                ) : (
                  <p className="profile-empty-text">
                    {isLoading ? 'Loading activity...' : 'No activity yet.'}
                  </p>
                )}
              </div>

              {activity.length > ACTIVITY_PAGE_SIZE ? (
                <div className="profile-activity-pagination" aria-label="Activity pages">
                  {Array.from({ length: totalActivityPages }, (_, index) => index + 1).map(
                    (page) => (
                      <button
                        key={page}
                        type="button"
                        className={`profile-activity-page-button ${
                          page === activityPage ? 'profile-activity-page-active' : ''
                        }`}
                        onClick={() => setActivityPage(page)}
                      >
                        {page}
                      </button>
                    ),
                  )}
                </div>
              ) : null}
            </article>
          </div>

          <aside className="profile-right-column">
            <ProfileCard icon={Users} title="Account Settings">
              <SettingsRow
                icon={Mail}
                label="Email Address"
                value={displayEmail}
                onClick={showComingSoon}
              />
              <SettingsRow
                icon={Globe}
                label="Language & Region"
                value={languageRegion}
                onClick={showComingSoon}
              />
            </ProfileCard>

            <ProfileCard icon={Shield} title="Security">
              <SettingsRow
                icon={Clock}
                label="Change Password"
                badge="90 days ago"
                onClick={showComingSoon}
              />
              <div className="profile-security-row">
                <CheckSquare className="profile-settings-icon" aria-hidden="true" />
                <span className="profile-settings-content">
                  <span className="profile-settings-label">2FA Verification</span>
                  <span className="profile-recommended-badge">RECOMMENDED</span>
                </span>
                <Toggle checked label="2FA Verification" onClick={showComingSoon} />
              </div>
            </ProfileCard>

            <ProfileCard icon={Bell} title="Notifications">
              <div className="profile-notification-list">
                <div className="profile-notification-row">
                  <span>
                    <span className="profile-settings-label">Critical Alerts</span>
                    <span className="profile-settings-value">
                      Immediate push to mobile and email
                    </span>
                  </span>
                  <Toggle
                    checked={notificationPrefs.critical_alerts}
                    label="Critical Alerts"
                    onClick={() => toggleNotification('critical_alerts')}
                  />
                </div>
                <div className="profile-notification-row">
                  <span>
                    <span className="profile-settings-label">Case Updates</span>
                    <span className="profile-settings-value">
                      Daily digest of reviewed materials
                    </span>
                  </span>
                  <Toggle
                    checked={notificationPrefs.case_updates}
                    label="Case Updates"
                    onClick={() => toggleNotification('case_updates')}
                  />
                </div>
                <div className="profile-notification-row">
                  <span>
                    <span className="profile-settings-label">Login History</span>
                    <span className="profile-settings-value">
                      Notify on new device sign-in
                    </span>
                  </span>
                  <Toggle
                    checked={notificationPrefs.login_history}
                    label="Login History"
                    onClick={() => toggleNotification('login_history')}
                  />
                </div>
              </div>
            </ProfileCard>

            <button type="button" className="profile-logout-button" onClick={handleLogout}>
              <LogOut aria-hidden="true" />
              Log Out from Session
            </button>
          </aside>
        </div>
      </div>
    </section>
  )
}
