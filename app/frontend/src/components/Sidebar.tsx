import { BarChart3, Bell, ListVideo, Plus, UploadCloud, User, Video } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { NavLink } from 'react-router-dom'

interface NavItem {
  label: string
  to: string
  icon: LucideIcon
}

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard', to: '/dashboard', icon: BarChart3 },
  { label: 'Upload Video', to: '/', icon: UploadCloud },
  { label: 'Queue Analyze', to: '/queue', icon: ListVideo },
  { label: 'Alerts', to: '/alerts', icon: Bell },
  { label: 'Profile', to: '/profile', icon: User },
]

export function Sidebar() {
  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-[#C3C6D7] bg-white px-4 py-4 lg:min-h-screen lg:w-60 lg:border-b-0 lg:border-r lg:bg-[#F2F3FF]">
      <div className="mb-4 flex items-center gap-4 px-2 lg:mb-6">
        <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#004AC6] text-white">
          <Video className="h-5 w-5" aria-hidden="true" />
        </span>
        <h1 className="text-base font-extrabold leading-6 text-[#131B2E]">
          Video Anomaly
          <br />
          Detection
        </h1>
      </div>

      <nav className="flex gap-1 overflow-x-auto lg:block lg:space-y-1" aria-label="Primary navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                [
                  'flex items-center gap-4 whitespace-nowrap rounded-lg px-4 py-2 text-xs uppercase tracking-wide transition',
                  isActive
                    ? 'bg-[#D0E1FB] font-bold text-[#131B2E]'
                    : 'font-medium text-[#434655] hover:bg-slate-100 hover:text-slate-950',
                ].join(' ')
              }
            >
              <Icon className="h-5 w-5 shrink-0" aria-hidden="true" />
              {item.label}
            </NavLink>
          )
        })}
      </nav>

      <div className="mt-4 border-t border-[#C3C6D7] pt-4 lg:mt-auto">
        <NavLink
          to="/"
          className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-[#004AC6] px-4 py-2 text-xs font-medium uppercase tracking-wide text-white shadow-md transition hover:opacity-90"
        >
          <Plus className="h-4 w-4" aria-hidden="true" />
          New Investigation
        </NavLink>
      </div>
    </aside>
  )
}
