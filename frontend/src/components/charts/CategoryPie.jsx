import { useMemo } from 'react'

const RADIUS = 38
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

const COLOR_INTERNAL = '#3b82f6' // Blue
const COLOR_CLIENT = '#a855f7'   // Purple

export default function CategoryPie({ meetings = [] }) {
  const data = useMemo(() => {
    let internal = 0
    let client = 0
    meetings.forEach((m) => {
      if (m.meetingType === 'client') client++
      else internal++
    })
    return [
      { name: 'Internal', value: internal, color: COLOR_INTERNAL, icon: '👥' },
      { name: 'Client', value: client, color: COLOR_CLIENT, icon: '🤝' },
    ]
  }, [meetings])

  const total = data.reduce((sum, d) => sum + d.value, 0)

  if (!total) {
    return (
      <div className="flex flex-col items-center justify-center py-6 text-text-muted font-meta-data text-xs">
        No meetings recorded yet
      </div>
    )
  }

  let offset = 0
  const segments = data.map((d) => {
    const length = total > 0 ? (d.value / total) * CIRCUMFERENCE : 0
    const seg = { ...d, length, offset: -offset }
    offset += length
    return seg
  })

  const internalSegment = data.find((d) => d.name === 'Internal') || { value: 0 }
  const internalPct = total > 0 ? Math.round((internalSegment.value / total) * 100) : 0

  return (
    <div className="flex flex-col sm:flex-row items-center justify-around gap-4 h-full">
      {/* Donut graphic */}
      <div className="relative w-32 h-32 shrink-0">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r={RADIUS}
            fill="transparent"
            stroke="rgb(var(--color-border))"
            strokeWidth="10"
          />
          {segments.map((s) =>
            s.length > 0 ? (
              <circle
                key={s.name}
                cx="50"
                cy="50"
                r={RADIUS}
                fill="transparent"
                stroke={s.color}
                strokeDasharray={`${s.length} ${CIRCUMFERENCE}`}
                strokeDashoffset={s.offset}
                strokeWidth="10"
                strokeLinecap="round"
                className="transition-all duration-500 ease-out hover:opacity-85"
              >
                <title>
                  {s.name}: {s.value} ({Math.round((s.value / total) * 100)}%)
                </title>
              </circle>
            ) : null
          )}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="font-headline-lg text-xl font-bold text-text-primary">{internalPct}%</span>
          <span className="font-meta-data text-[10px] text-text-muted uppercase tracking-wider">Internal</span>
        </div>
      </div>

      {/* Legend list */}
      <div className="flex flex-col gap-2.5 min-w-[130px]">
        {data.map((d) => {
          const pct = total > 0 ? Math.round((d.value / total) * 100) : 0
          return (
            <div key={d.name} className="flex items-center justify-between gap-3 text-xs bg-surface-raised/60 p-2 rounded-lg border border-border/50">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="font-meta-data text-text-primary font-medium">{d.icon} {d.name}</span>
              </div>
              <span className="font-mono text-text-muted font-bold">{d.value} <span className="text-[10px] font-normal text-text-faint">({pct}%)</span></span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
