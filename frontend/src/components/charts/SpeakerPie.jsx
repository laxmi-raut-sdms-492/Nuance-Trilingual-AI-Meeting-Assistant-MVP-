/**
 * Talk-time donut, ported from the design export (dashboard_populated).
 *
 * SVG stroke-dasharray segments as in the export, not Recharts. Segment
 * colours come from the backend (`speakerStats[].color`, assigned in
 * first-appearance order) so a speaker keeps one colour across the transcript,
 * the bars and this chart. Never re-map them here.
 */
const RADIUS = 40
const CIRCUMFERENCE = 2 * Math.PI * RADIUS // ≈ 251.3

export default function SpeakerPie({ data }) {
  const total = data.reduce((sum, d) => sum + d.value, 0)
  if (!total) return null

  let offset = 0
  const segments = data.map((d) => {
    const length = (d.value / total) * CIRCUMFERENCE
    const seg = { ...d, length, offset: -offset }
    offset += length
    return seg
  })

  const top = segments[0]
  const topPct = Math.round((top.value / total) * 100)

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative w-40 h-40">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r={RADIUS}
            fill="transparent"
            stroke="rgb(var(--color-border))"
            strokeWidth="12"
          />
          {segments.map((s) => (
            <circle
              key={s.name}
              className="donut-segment"
              cx="50"
              cy="50"
              r={RADIUS}
              fill="transparent"
              stroke={s.color}
              strokeDasharray={`${s.length} ${CIRCUMFERENCE}`}
              strokeDashoffset={s.offset}
              strokeWidth="12"
            >
              <title>
                {s.name}: {Math.round((s.value / total) * 100)}%
              </title>
            </circle>
          ))}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-headline-lg text-headline-lg text-text-primary">{topPct}%</span>
          <span className="font-label-sm text-label-sm text-text-muted truncate max-w-[100px]">
            {top.name}
          </span>
        </div>
      </div>

      <div className="flex gap-4 mt-6 w-full justify-center flex-wrap">
        {segments.slice(0, 5).map((s) => (
          <div key={s.name} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
            <span className="font-meta-data text-meta-data text-text-muted">{s.name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
