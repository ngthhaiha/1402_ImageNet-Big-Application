interface LoadingSpinnerProps {
  label?: string
  size?: 'sm' | 'md' | 'lg'
}

const SIZE_CLASS_NAMES: Record<NonNullable<LoadingSpinnerProps['size']>, string> = {
  sm: 'h-4 w-4',
  md: 'h-6 w-6',
  lg: 'h-8 w-8',
}

export function LoadingSpinner({ label = 'Loading', size = 'md' }: LoadingSpinnerProps) {
  return (
    <div className="inline-flex items-center gap-2 text-sm font-medium text-slate-600">
      <span
        className={`${SIZE_CLASS_NAMES[size]} animate-spin rounded-full border-2 border-slate-300 border-t-blue-600`}
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  )
}
