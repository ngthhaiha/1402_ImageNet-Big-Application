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
import type {
  AlertFilter,
  AlertLogItem,
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

function escapeCsvCell(value: string | number | null): string {
  const stringValue = value === null ? '' : String(value)
  if (/[",\n\r]/.test(stringValue)) {
    return `"${stringValue.replaceAll('"', '""')}"`
  }

  return stringValue
}

function buildAlertLogCsv(items: AlertLogItem[]): string {
  const headers = [
    'Time',
    'Video Name',
    'Activity Type',
    'Confidence',
    'Severity',
    'Status',
    'Start Time',
    'End Time',
    'Anomaly Score',
  ]
  const rows = items.map((item) => [
    item.time,
    item.filename,
    item.activity_type,
    `${(item.confidence_score * 100).toFixed(1)}%`,
    item.severity,
    item.status,
    item.start_time,
    item.end_time,
    item.anomaly_score.toFixed(3),
  ])

  return [headers, ...rows]
    .map((row) => row.map(escapeCsvCell).join(','))
    .join('\n')
}

function downloadCsv(filename: string, csvContent: string) {
  const blob = new Blob([`\uFEFF${csvContent}`], {
    type: 'text/csv;charset=utf-8;',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
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
  const [isExportingLog, setIsExportingLog] = useState(false)
  const [isFilterVisible, setIsFilterVisible] = useState(false)
  const [error, setError] = useState<string | null>(null)

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

  const handleExportAlertLog = useCallback(async () => {
    setIsExportingLog(true)
    try {
      setError(null)
      const firstResponse = assertData(await getAlertLog(filter, 1, 100))
      const allItems = [...firstResponse.items]
      for (let nextPage = 2; nextPage <= firstResponse.total_pages; nextPage += 1) {
        const response = assertData(await getAlertLog(filter, nextPage, 100))
        allItems.push(...response.items)
      }

      const csv = buildAlertLogCsv(allItems)
      const today = new Date().toISOString().slice(0, 10)
      downloadCsv(`alert-log-${today}.csv`, csv)
    } catch (exportError) {
      setError(getErrorMessage(exportError))
    } finally {
      setIsExportingLog(false)
    }
  }, [filter])

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
      <PageHeader pageName="Alert" />

      <div className="alerts-shell">
        {error ? <div className="alerts-error">{error}</div> : null}

        <section className="alerts-stats-grid" aria-label="Alert summary">
          {statCards.map((card) => (
            <AlertStatsCard key={card.label} {...card} />
          ))}
        </section>

        {isFilterVisible ? (
          <AlertFilterBar value={filter} onChange={handleFilterChange} />
        ) : null}

        <section className="alerts-main-row">
          <div className="alerts-log-column">
            <AlertLogTable
              data={logData}
              isLoading={isLogLoading}
              page={page}
              onPageChange={setPage}
              onViewInvestigation={handleViewSegment}
              onToggleFilter={() => setIsFilterVisible((current) => !current)}
              onExportCsv={() => void handleExportAlertLog()}
              isFilterVisible={isFilterVisible}
              isExporting={isExportingLog}
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
