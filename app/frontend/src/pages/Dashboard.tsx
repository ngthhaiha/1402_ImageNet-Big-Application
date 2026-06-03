import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CheckCircle2, Cloud, ShieldAlert, Video } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import {
  getDashboardDistribution,
  getDashboardRecentActivity,
  getDashboardRecentAlerts,
  getDashboardRecentInvestigations,
  getDashboardStats,
  getDashboardTopDetections,
} from '../api/api'
import { AnomalyDonut } from '../components/dashboard/AnomalyDonut'
import { DashboardFilter } from '../components/dashboard/DashboardFilter'
import { RecentActivity } from '../components/dashboard/RecentActivity'
import { RecentAlerts } from '../components/dashboard/RecentAlerts'
import { RecentInvestigations } from '../components/dashboard/RecentInvestigations'
import { StatsCard } from '../components/dashboard/StatsCard'
import { TopDetections } from '../components/dashboard/TopDetections'
import { Toast } from '../components/Toast'
import type {
  ActivityItem,
  AlertItem,
  DashboardFilter as DashboardFilterState,
  DashboardStats,
  DistributionItem,
  InvestigationItem,
  TopDetection,
} from '../types/types'

const EMPTY_FILTER: DashboardFilterState = {
  anomaly_class: '',
  date_from: '',
  date_to: '',
}

const INVESTIGATION_PAGE_SIZE = 5

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unable to load dashboard data'
}

function assertData<T>(response: { success: boolean; data: T | null; message: string }): T {
  if (!response.success || response.data === null) {
    throw new Error(response.message)
  }

  return response.data
}

function takeInvestigationPage(items: InvestigationItem[]): {
  pageItems: InvestigationItem[]
  hasMore: boolean
} {
  return {
    pageItems: items.slice(0, INVESTIGATION_PAGE_SIZE),
    hasMore: items.length > INVESTIGATION_PAGE_SIZE,
  }
}

export function Dashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [distribution, setDistribution] = useState<DistributionItem[]>([])
  const [activity, setActivity] = useState<ActivityItem[]>([])
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [topDetections, setTopDetections] = useState<TopDetection[]>([])
  const [investigations, setInvestigations] = useState<InvestigationItem[]>([])
  const [hasMoreInvestigations, setHasMoreInvestigations] = useState(false)
  const [filter, setFilter] = useState<DashboardFilterState>(EMPTY_FILTER)
  const [isFilterOpen, setIsFilterOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toastOpen, setToastOpen] = useState(false)

  const totalInvestigationCount = hasMoreInvestigations
    ? investigations.length + 1
    : investigations.length

  const statCards = useMemo(
    () => [
      {
        icon: Video,
        iconColor: 'text-[#004AC6]',
        iconBg: 'stats-icon-bg-blue',
        badge: '+12%',
        badgeColor: 'text-[#004AC6]',
        badgeBg: 'stats-badge-bg-blue',
        value: stats?.total_videos ?? 0,
        valueColor: 'text-[#131B2E]',
        label: 'Total Videos Analyzed',
        description: 'Total uploaded videos processed by AI',
      },
      {
        icon: AlertTriangle,
        iconColor: 'text-[#BA1A1A]',
        iconBg: 'stats-icon-bg-red',
        badge: '+3%',
        badgeColor: 'text-[#BA1A1A]',
        badgeBg: 'stats-badge-bg-red',
        value: stats?.total_anomalies ?? 0,
        valueColor: 'text-[#BA1A1A]',
        label: 'Abnormal Events Detected',
        description: 'Total abnormal segments detected',
      },
      {
        icon: ShieldAlert,
        iconColor: 'text-[#943700]',
        iconBg: 'stats-icon-bg-brown',
        badge: '24 New',
        badgeColor: 'text-[#943700]',
        badgeBg: 'stats-badge-bg-brown',
        value: stats?.pending_reviews ?? 0,
        valueColor: 'text-[#131B2E]',
        label: 'Pending Reviews',
        description: 'Segments waiting for user validation',
      },
      {
        icon: CheckCircle2,
        iconColor: 'text-[#505F76]',
        iconBg: 'stats-icon-bg-gray',
        badge: '98% Acc.',
        badgeColor: 'text-[#505F76]',
        badgeBg: 'stats-badge-bg-gray',
        value: stats?.reviewed_cases ?? 0,
        valueColor: 'text-[#131B2E]',
        label: 'Reviewed Cases',
        description: 'Validated anomaly events',
      },
    ],
    [stats],
  )

  useEffect(() => {
    let isMounted = true

    async function loadUnfilteredData() {
      try {
        setError(null)
        const [statsResponse, distributionResponse, activityResponse] = await Promise.all([
          getDashboardStats(),
          getDashboardDistribution(),
          getDashboardRecentActivity(4),
        ])

        if (!isMounted) {
          return
        }

        setStats(assertData(statsResponse))
        setDistribution(assertData(distributionResponse))
        setActivity(assertData(activityResponse))
      } catch (loadError) {
        if (isMounted) {
          setError(getErrorMessage(loadError))
        }
      }
    }

    void loadUnfilteredData()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    let isMounted = true

    async function loadFilteredData() {
      setIsLoading(true)
      try {
        setError(null)
        const [alertsResponse, topDetectionsResponse, investigationsResponse] = await Promise.all([
          getDashboardRecentAlerts(filter, 4),
          getDashboardTopDetections(filter),
          getDashboardRecentInvestigations(filter, INVESTIGATION_PAGE_SIZE + 1, 0),
        ])

        if (!isMounted) {
          return
        }

        const nextInvestigationPage = takeInvestigationPage(assertData(investigationsResponse))
        setAlerts(assertData(alertsResponse))
        setTopDetections(assertData(topDetectionsResponse))
        setInvestigations(nextInvestigationPage.pageItems)
        setHasMoreInvestigations(nextInvestigationPage.hasMore)
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

    void loadFilteredData()

    return () => {
      isMounted = false
    }
  }, [filter])

  async function loadMoreInvestigations() {
    try {
      const response = await getDashboardRecentInvestigations(
        filter,
        INVESTIGATION_PAGE_SIZE + 1,
        investigations.length,
      )
      const nextInvestigationPage = takeInvestigationPage(assertData(response))
      setInvestigations((current) => [...current, ...nextInvestigationPage.pageItems])
      setHasMoreInvestigations(nextInvestigationPage.hasMore)
    } catch (loadError) {
      setError(getErrorMessage(loadError))
    }
  }

  function showComingSoon() {
    setToastOpen(true)
  }

  return (
    <section className="min-h-screen bg-[#FAF8FF] px-8 py-8 text-[#131B2E]">
      <Toast
        open={toastOpen}
        title="Coming soon"
        message="This dashboard action is not available in the demo yet."
        variant="info"
        onClose={() => setToastOpen(false)}
      />

      <div className="mx-auto flex w-full max-w-7xl flex-col gap-8">
        <section className="relative overflow-hidden rounded-xl border border-[rgba(195,198,215,0.30)] bg-white p-12 shadow-sm">
          {/* Decorative blur circle — matches Figma */}
          <div
            className="pointer-events-none absolute"
            style={{
              width: 256,
              height: 256,
              right: -30,
              top: -63,
              background: 'rgba(0, 74, 198, 0.05)',
              borderRadius: 9999,
              filter: 'blur(32px)',
            }}
          />
          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-col gap-2">
              <h1 className="dashboard-welcome-title text-[#131B2E]">Welcome back</h1>
              <p className="dashboard-welcome-subtitle max-w-xl font-normal text-[#434655]">
                Monitor abnormal activities detected from surveillance videos and review
                investigation results.
              </p>
            </div>
            <button
              type="button"
              className="relative inline-flex shrink-0 items-center justify-center gap-4 rounded-lg bg-[#004AC6] px-8 py-4 text-base font-normal text-white hover:opacity-90"
              style={{
                boxShadow:
                  '0px 4px 6px -4px rgba(0, 74, 198, 0.20), 0px 10px 15px -3px rgba(0, 74, 198, 0.20)',
              }}
              onClick={() => navigate('/')}
            >
              <Cloud className="h-5 w-5" aria-hidden="true" />
              Upload New Video
            </button>
          </div>
        </section>

        {error ? (
          <div className="rounded-xl border border-[rgba(195,198,215,0.30)] bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
          {statCards.map((card) => (
            <StatsCard key={card.label} {...card} />
          ))}
        </div>

        <section className="flex flex-col gap-6 xl:flex-row">
          <div className="flex-1">
            <AnomalyDonut items={distribution} />
          </div>
          <div className="w-full shrink-0 xl:w-80">
            <RecentActivity items={activity} />
          </div>
        </section>

        <section className="flex flex-col gap-6 xl:flex-row">
          <div className="flex-1">
            <RecentAlerts
              items={alerts}
              onRowClick={(videoId) => navigate(`/videos/${videoId}`)}
              onViewAll={showComingSoon}
            />
          </div>
          <div className="w-full shrink-0 xl:w-80">
            <TopDetections items={topDetections} />
          </div>
        </section>

        <RecentInvestigations
          items={investigations}
          totalCount={totalInvestigationCount}
          filterPanel={
            isFilterOpen ? (
              <DashboardFilter
                value={filter}
                onFilterChange={(nextFilter) => {
                  setFilter(nextFilter)
                  setIsFilterOpen(false)
                }}
              />
            ) : null
          }
          onLoadMore={loadMoreInvestigations}
          onRowClick={(videoId) => navigate(`/videos/${videoId}`)}
          onFilterClick={() => setIsFilterOpen((current) => !current)}
          onExportData={showComingSoon}
        />

        {isLoading ? (
          <p className="text-center text-sm font-medium text-[#505F76]">Loading dashboard data...</p>
        ) : null}
      </div>
    </section>
  )
}
