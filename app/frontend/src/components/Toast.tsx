type ToastVariant = 'success' | 'error' | 'info'

interface ToastProps {
  open: boolean
  title: string
  message?: string
  variant?: ToastVariant
  onClose?: () => void
}

const VARIANT_CLASS_NAMES: Record<ToastVariant, string> = {
  success: 'border-emerald-200 bg-emerald-50 text-emerald-950',
  error: 'border-red-200 bg-red-50 text-red-950',
  info: 'border-blue-200 bg-blue-50 text-blue-950',
}

export function Toast({ open, title, message, variant = 'info', onClose }: ToastProps) {
  if (!open) {
    return null
  }

  return (
    <div
      className={`fixed left-4 right-4 top-4 z-50 max-w-sm rounded-lg border p-4 shadow-lg sm:left-auto ${VARIANT_CLASS_NAMES[variant]}`}
      role="status"
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold">{title}</p>
          {message ? <p className="mt-1 text-sm opacity-80">{message}</p> : null}
        </div>
        {onClose ? (
          <button
            type="button"
            className="rounded p-1 text-sm font-semibold opacity-70 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-current"
            aria-label="Dismiss toast"
            onClick={onClose}
          >
            x
          </button>
        ) : null}
      </div>
    </div>
  )
}
