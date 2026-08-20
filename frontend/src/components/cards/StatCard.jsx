import Icon from '../common/Icon.jsx'

/**
 * Clean, non-clickable KPI StatCard component.
 * Displays metrics cleanly with no pointer cursor or click handlers.
 */
const tints = {
  primary: { value: 'text-text-primary', icon: 'text-text-muted' },
  green: { value: 'text-success', icon: 'text-success/70' },
  amber: { value: 'text-processing', icon: 'text-processing/70' },
  red: { value: 'text-error', icon: 'text-error/70' },
}

export default function StatCard({ label, value, icon, tint = 'primary' }) {
  const t = tints[tint] || tints.primary
  const isLongValue = String(value).length > 4

  return (
    <div className="bg-surface border border-border rounded-xl p-3.5 sm:p-4 flex flex-col justify-between relative overflow-hidden text-left w-full cursor-default">
      <div className="flex justify-between items-start relative z-10 gap-1.5 w-full">
        <div className="min-w-0">
          <p className="font-label-sm text-[10px] sm:text-[11px] text-text-muted uppercase tracking-wider mb-1 truncate">
            {label}
          </p>
          <p className={`${isLongValue ? 'text-base sm:text-lg font-bold' : 'text-lg sm:text-2xl font-bold'} ${t.value} truncate`}>
            {value}
          </p>
        </div>
        <div
          className={`w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-surface-raised border border-border flex items-center justify-center shrink-0 ${t.icon}`}
        >
          <Icon name={icon} size={16} />
        </div>
      </div>
    </div>
  )
}
