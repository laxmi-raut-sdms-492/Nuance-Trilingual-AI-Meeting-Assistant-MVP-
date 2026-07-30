import Icon from './Icon.jsx'

/**
 * Empty states are a deliverable here, not a fallback: the stand demo starts
 * with zero data, and several panels are empty by design — trash and
 * integrations have no backing, and the summarization panels stay empty
 * whenever nothing survived citation verification. They must look designed,
 * never like a bug.
 *
 * The dashed ring is the export's `empty-dash` animation.
 */
export default function EmptyState({
  title = 'Nothing here yet',
  subtitle = '',
  icon = 'inbox',
  action = null,
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <div className="relative w-16 h-16 mb-4 flex items-center justify-center">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 64 64" aria-hidden="true">
          <circle
            className="empty-dash"
            cx="32"
            cy="32"
            r="30"
            fill="none"
            stroke="rgb(var(--color-border))"
            strokeWidth="2"
          />
        </svg>
        <Icon name={icon} size={24} className="text-text-muted" />
      </div>
      <p className="text-text-primary font-semibold">{title}</p>
      {subtitle && (
        <p className="font-meta-data text-meta-data text-text-muted mt-1.5 max-w-sm">{subtitle}</p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}
