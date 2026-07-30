import Icon from '../common/Icon.jsx'

/**
 * Ported from the dashboard stat grid in the design export.
 *
 * The value inherits the tint (Completed reads success-green, Processing
 * reads amber) and the icon well brightens on hover, both as in the export.
 * `stat-card-hover` supplies the lift and orange glow — see index.css.
 */
const tints = {
  primary: { value: 'text-text-primary', icon: 'text-text-muted group-hover:text-primary' },
  green: { value: 'text-success', icon: 'text-success/70 group-hover:text-success' },
  amber: { value: 'text-processing', icon: 'text-processing/70 group-hover:text-processing' },
  red: { value: 'text-error', icon: 'text-error/70 group-hover:text-error' },
}

export default function StatCard({ label, value, icon, tint = 'primary' }) {
  const t = tints[tint] || tints.primary

  return (
    <div className="bg-surface border border-border rounded-xl p-5 flex flex-col justify-between stat-card-hover transition-all relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-to-br from-surface-raised to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="flex justify-between items-start relative z-10">
        <div>
          <p className="font-label-sm text-label-sm text-text-muted uppercase tracking-wider mb-2">
            {label}
          </p>
          <p className={`font-headline-lg text-headline-lg ${t.value}`}>{value}</p>
        </div>
        <div
          className={`w-10 h-10 rounded-lg bg-surface-raised border border-border flex items-center justify-center transition-colors ${t.icon}`}
        >
          <Icon name={icon} />
        </div>
      </div>
    </div>
  )
}
