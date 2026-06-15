import { useEffect, useState } from 'react'
import { ChevronDown, Search } from 'lucide-react'

import { ANOMALY_LABELS } from '../../types/types'
import type { AlertFilter } from '../../types/types'

interface AlertFilterBarProps {
  value: AlertFilter
  onChange: (filter: AlertFilter) => void
}

const ACTIVITY_OPTIONS = ANOMALY_LABELS.filter((label) => label !== 'Normal' && label !== 'Other')

export function AlertFilterBar({ value, onChange }: AlertFilterBarProps) {
  const [nameInput, setNameInput] = useState(value.name)

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      if (nameInput !== value.name) {
        onChange({ ...value, name: nameInput })
      }
    }, 300)

    return () => window.clearTimeout(timeoutId)
  }, [nameInput, onChange, value])

  function updateFilter(nextFilter: AlertFilter) {
    onChange(nextFilter)
  }

  function resetFilters() {
    setNameInput('')
    onChange({
      name: '',
      activity: '',
      severity: '',
      status: '',
      date: '',
    })
  }

  return (
    <section className="alerts-card alerts-filter-bar" aria-label="Alert filters">
      <label className="alerts-filter-search">
        <Search className="alerts-filter-icon" aria-hidden="true" />
        <input
          type="text"
          value={nameInput}
          onChange={(event) => setNameInput(event.target.value)}
          placeholder="Filter by video name..."
        />
      </label>

      <label className="alerts-select-wrap">
        <select
          value={value.activity}
          onChange={(event) => updateFilter({ ...value, activity: event.target.value })}
          aria-label="Activity Type"
        >
          <option value="">All</option>
          {ACTIVITY_OPTIONS.map((label) => (
            <option key={label} value={label}>
              {label}
            </option>
          ))}
        </select>
        <ChevronDown className="alerts-select-icon" aria-hidden="true" />
      </label>

      <label className="alerts-select-wrap">
        <select
          value={value.severity}
          onChange={(event) =>
            updateFilter({ ...value, severity: event.target.value as AlertFilter['severity'] })
          }
          aria-label="Severity"
        >
          <option value="">All</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
        </select>
        <ChevronDown className="alerts-select-icon" aria-hidden="true" />
      </label>

      <label className="alerts-select-wrap">
        <select
          value={value.status}
          onChange={(event) =>
            updateFilter({ ...value, status: event.target.value as AlertFilter['status'] })
          }
          aria-label="Review Status"
        >
          <option value="">All</option>
          <option value="PENDING_REVIEW">Unreviewed</option>
          <option value="REVIEWED">Reviewed</option>
        </select>
        <ChevronDown className="alerts-select-icon" aria-hidden="true" />
      </label>

      <input
        className="alerts-date-input"
        type="date"
        value={value.date}
        onChange={(event) => updateFilter({ ...value, date: event.target.value })}
        aria-label="Alert date"
      />

      <button type="button" className="alerts-reset-button" onClick={resetFilters}>
        Reset Filters
      </button>
    </section>
  )
}
