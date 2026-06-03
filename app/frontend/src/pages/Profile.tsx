import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
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
import { Toast } from '../components/Toast'
import type {
  DashboardActivityType,
  ProfileActivityItem,
  ProfileStats,
} from '../types/types'

const NOTIFICATION_STORAGE_KEY = 'notification_prefs'

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
  const [stats, setStats] = useState<ProfileStats | null>(null)
  const [activity, setActivity] = useState<ProfileActivityItem[]>([])
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

  useEffect(() => {
    let isMounted = true

    async function loadProfileData() {
      setIsLoading(true)
      try {
        setError(null)
        const [statsResponse, activityResponse] = await Promise.all([
          getProfileStats(),
          getProfileActivity(10),
        ])

        if (!isMounted) {
          return
        }

        setStats(assertData(statsResponse))
        setActivity(assertData(activityResponse))
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

      <div className="profile-shell">
        <section className="profile-header-card">
          <div className="profile-identity">
            <div className="profile-avatar-wrap">
              <div className="profile-avatar" aria-label="Officer James Miller avatar">
                JM
              </div>
              <span className="profile-camera-badge">
                <Camera aria-hidden="true" />
              </span>
            </div>

            <div className="profile-identity-text">
              <h1>Officer James Miller</h1>
              <p>Senior Security Investigator</p>
              <div className="profile-badges">
                <span className="profile-badge profile-badge-id">ID: #SOC-882</span>
                <span className="profile-badge profile-badge-location">Sector A, London</span>
                <span className="profile-badge profile-badge-shift">14h Current Shift</span>
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
                  activity.map((item) => {
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
                            {item.description ?? 'No additional details'}
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

              <button type="button" className="profile-history-button" onClick={showComingSoon}>
                View Full Activity History
              </button>
            </article>
          </div>

          <aside className="profile-right-column">
            <ProfileCard icon={Users} title="Account Settings">
              <SettingsRow
                icon={Mail}
                label="Email Address"
                value="j.miller@ssis.hq.com"
                onClick={showComingSoon}
              />
              <SettingsRow
                icon={Globe}
                label="Language & Region"
                value="English (UK) - UTC +0"
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

            <button type="button" className="profile-logout-button" onClick={showComingSoon}>
              <LogOut aria-hidden="true" />
              Log Out from Session
            </button>
          </aside>
        </div>
      </div>
    </section>
  )
}
