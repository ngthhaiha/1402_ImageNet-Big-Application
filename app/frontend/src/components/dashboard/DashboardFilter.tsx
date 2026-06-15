import { useEffect, useState } from 'react'

import { ANOMALY_LABELS, type DashboardFilter as DashboardFilterState } from '../../types/types'

interface DashboardFilterProps {
  value: DashboardFilterState
  onFilterChange: (filter: DashboardFilterState) => void
}

const EMPTY_FILTER: DashboardFilterState = {
  anomaly_class: '',
  date_from: '',
  date_to: '',
}

export function DashboardFilter({ value, onFilterChange }: DashboardFilterProps) {
  const [draft, setDraft] = useState<DashboardFilterState>(value)

  useEffect(() => {
    setDraft(value)
  }, [value])

  function applyFilter() {
    onFilterChange(draft)
  }

  function resetFilter() {
    setDraft(EMPTY_FILTER)
    onFilterChange(EMPTY_FILTER)
  }

  return (
    <div className="border-y border-[#C3C6D7] bg-[rgba(242,243,255,0.35)] px-6 py-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-[1.2fr_1fr_1fr_auto_auto] xl:items-end">
        <label className="flex flex-col gap-2 text-sm font-semibold text-[#434655]">
          Anomaly Class
          <select
            className="h-11 rounded-lg border border-[#C3C6D7] bg-white px-3 text-sm font-medium text-[#131B2E] outline-none focus:border-[#C3C6D7] focus:ring-2 focus:ring-[#E2E7FF]"
            value={draft.anomaly_class}
            onChange={(event) =>
              setDraft((current) => ({ ...current, anomaly_class: event.target.value }))
            }
          >
            <option value="">All Classes</option>
            {ANOMALY_LABELS.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm font-semibold text-[#434655]">
          Date From
          <input
            className="h-11 rounded-lg border border-[#C3C6D7] bg-white px-3 text-sm font-medium text-[#131B2E] outline-none focus:border-[#C3C6D7] focus:ring-2 focus:ring-[#E2E7FF]"
            type="date"
            value={draft.date_from}
            onChange={(event) => setDraft((current) => ({ ...current, date_from: event.target.value }))}
          />
        </label>

        <label className="flex flex-col gap-2 text-sm font-semibold text-[#434655]">
          Date To
          <input
            className="h-11 rounded-lg border border-[#C3C6D7] bg-white px-3 text-sm font-medium text-[#131B2E] outline-none focus:border-[#C3C6D7] focus:ring-2 focus:ring-[#E2E7FF]"
            type="date"
            value={draft.date_to}
            onChange={(event) => setDraft((current) => ({ ...current, date_to: event.target.value }))}
          />
        </label>

        <button
          type="button"
          className="inline-flex h-11 items-center justify-center rounded-lg bg-[#004AC6] px-5 text-sm font-semibold text-white hover:opacity-90"
          onClick={applyFilter}
        >
          Apply
        </button>
        <button
          type="button"
          className="inline-flex h-11 items-center justify-center rounded-lg border border-[#C3C6D7] bg-white px-5 text-sm font-semibold text-[#505F76] hover:bg-gray-50"
          onClick={resetFilter}
        >
          Reset
        </button>
      </div>
    </div>
  )
}
