import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  BarChart3,
  CheckCircle,
  ClipboardList,
  TrendingUp,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import {
  getAlertDistribution,
  getAlertLog,
  getAlertStats,
  getCriticalAlerts,
} from '../api/api'
import { AlertDistribution } from '../components/alerts/AlertDistribution'
import { AlertFilterBar } from '../components/alerts/AlertFilterBar'
import { AlertLogTable } from '../components/alerts/AlertLogTable'
import { AlertStatsCard } from '../components/alerts/AlertStatsCard'
import { CriticalAlertsTable } from '../components/alerts/CriticalAlertsTable'
import { PageHeader } from '../components/PageHeader'
import { Toast } from '../components/Toast'
import type {
  AlertFilter,
  AlertLogResponse,
  AlertStats,
  CriticalAlertItem,
  DistributionItem,
} from '../types/types'

const EMPTY_ALERT_FILTER: AlertFilter = {
  name: '',
  activity: '',
  severity: '',
  status: '',
  date: '',
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unable to load alerts data'
}

function assertData<T>(response: { success: boolean; data: T | null; message: string }): T {
  if (!response.success || response.data === null) {
    throw new Error(response.message)
  }

  return response.data
}

export function Alerts() {
  const navigate = useNavigate()
  const [stats, setStats] = useState<AlertStats | null>(null)
  const [filter, setFilter] = useState<AlertFilter>(EMPTY_ALERT_FILTER)
  const [logData, setLogData] = useState<AlertLogResponse | null>(null)
  const [distribution, setDistribution] = useState<DistributionItem[]>([])
  const [criticalAlerts, setCriticalAlerts] = useState<CriticalAlertItem[]>([])
  const [page, setPage] = useState(1)
  const [isLogLoading, setIsLogLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toastOpen, setToastOpen] = useState(false)

  const statCards = useMemo(
    () => [
      {
        label: 'TOTAL ALERTS',
        value: stats?.total_alerts ?? 0,
        subText: '+12% from last week',
        subColor: '#16A34A',
        icon: BarChart3,
        trendIcon: TrendingUp,
        iconColor: '#004AC6',
      },
      {
        label: 'HIGH SEVERITY',
        value: stats?.high_severity ?? 0,
        subText: '5 active now',
        subColor: '#BA1A1A',
        icon: AlertCircle,
        trendIcon: AlertTriangle,
        iconColor: '#BA1A1A',
      },
      {
        label: 'PENDING REVIEWS',
        value: stats?.pending_reviews ?? 0,
        subText: 'Awaiting human validation',
        subColor: '#434655',
        icon: ClipboardList,
        iconColor: '#505F76',
      },
      {
        label: 'REVIEWED ALERTS',
        value: stats?.reviewed_alerts ?? 0,
        subText: '95.3% accuracy rate',
        subColor: '#004AC6',
        icon: CheckCircle,
        iconColor: '#004AC6',
      },
    ],
    [stats],
  )

  const showComingSoon = useCallback(() => {
    setToastOpen(true)
  }, [])

  const handleFilterChange = useCallback((nextFilter: AlertFilter) => {
    setFilter(nextFilter)
    setPage(1)
  }, [])

  const handleViewSegment = useCallback(
    (videoId: string, segmentId: number) => {
      navigate(`/videos/${videoId}?segment=${segmentId}`)
    },
    [navigate],
  )

  useEffect(() => {
    let isMounted = true

    async function loadStaticAlertsData() {
      try {
        setError(null)
        const [statsResponse, distributionResponse, criticalResponse] = await Promise.all([
          getAlertStats(),
          getAlertDistribution(),
          getCriticalAlerts(10),
        ])

        if (!isMounted) {
          return
        }

        setStats(assertData(statsResponse))
        setDistribution(assertData(distributionResponse))
        setCriticalAlerts(assertData(criticalResponse))
      } catch (loadError) {
        if (isMounted) {
          setError(getErrorMessage(loadError))
        }
      }
    }

    void loadStaticAlertsData()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    let isMounted = true

    async function loadAlertLog() {
      setIsLogLoading(true)
      try {
        setError(null)
        const response = await getAlertLog(filter, page, 10)
        if (!isMounted) {
          return
        }
        setLogData(assertData(response))
      } catch (loadError) {
        if (isMounted) {
          setError(getErrorMessage(loadError))
        }
      } finally {
        if (isMounted) {
          setIsLogLoading(false)
        }
      }
    }

    void loadAlertLog()

    return () => {
      isMounted = false
    }
  }, [filter, page])

  return (
    <section className="alerts-page">
      <Toast
        open={toastOpen}
        title="Coming soon"
        message="This alerts action is not available in the demo yet."
        variant="info"
        onClose={() => setToastOpen(false)}
      />

      <PageHeader pageName="Alert" />

      <div className="alerts-shell">
        {error ? <div className="alerts-error">{error}</div> : null}

        <section className="alerts-stats-grid" aria-label="Alert summary">
          {statCards.map((card) => (
            <AlertStatsCard key={card.label} {...card} />
          ))}
        </section>

        <AlertFilterBar value={filter} onChange={handleFilterChange} />

        <section className="alerts-main-row">
          <div className="alerts-log-column">
            <AlertLogTable
              data={logData}
              isLoading={isLogLoading}
              page={page}
              onPageChange={setPage}
              onViewInvestigation={handleViewSegment}
              onComingSoon={showComingSoon}
            />
          </div>
          <aside className="alerts-side-column">
            <AlertDistribution items={distribution} />
          </aside>
        </section>

        <CriticalAlertsTable
          items={criticalAlerts}
          onViewDetail={handleViewSegment}
        />
      </div>
    </section>
  )
}
