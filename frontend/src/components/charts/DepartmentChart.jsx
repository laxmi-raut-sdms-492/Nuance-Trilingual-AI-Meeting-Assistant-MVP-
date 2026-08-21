import { useMemo } from 'react'

const DEPT_ICONS = {
  'AI Team': '',
  'Software': '',
  'QA': '',
  'Product & Design': '',
  'Management': '',
  'Sales & Marketing': '',
}

const BAR_COLORS = [
  'bg-blue-500',
  'bg-purple-500',
  'bg-emerald-500',
  'bg-amber-500',
  'bg-rose-500',
  'bg-indigo-500',
]

export default function DepartmentChart({ meetings = [] }) {
  const depts = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      const d = m.department || 'AI Team'
      totals[d] = (totals[d] || 0) + 1
    })
    return Object.entries(totals)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
  }, [meetings])

  const maxVal = useMemo(() => Math.max(1, ...depts.map((d) => d.value)), [depts])

  if (!depts.length) {
    return (
      <div className="flex flex-col items-center justify-center py-6 text-text-muted font-meta-data text-xs">
        No department data recorded
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3 py-1">
      {depts.slice(0, 5).map((d, idx) => {
        const pct = Math.round((d.value / maxVal) * 100)
        const icon = DEPT_ICONS[d.name] || ''
        const barColor = BAR_COLORS[idx % BAR_COLORS.length]

        return (
          <div key={d.name} className="flex flex-col gap-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-meta-data text-text-primary font-medium flex items-center gap-1.5 truncate">
                <span>{icon}</span>
                <span className="truncate">{d.name}</span>
              </span>
              <span className="font-mono text-text-muted font-bold text-xs">{d.value} {d.value === 1 ? 'meeting' : 'meetings'}</span>
            </div>
            <div className="w-full bg-surface-raised border border-border/40 h-2.5 rounded-full overflow-hidden">
              <div
                className={`h-full ${barColor} rounded-full transition-all duration-500 ease-out`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
