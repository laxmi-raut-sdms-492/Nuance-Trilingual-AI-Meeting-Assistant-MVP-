/**
 * Weekly activity bars, ported from the design export (dashboard_populated).
 *
 * Built from divs rather than Recharts, exactly as the export does. That is
 * both the faithful port and the simpler one — Recharts defaults are
 * unreadable on a near-black ground and would need every axis, grid, tooltip
 * and legend colour overridden by hand.
 *
 * Three departures from the export's static markup, all readability fixes
 * found once real data went through it:
 *
 *  - Every day gets a full-height **channel** behind its bar. The export only
 *    draws the bar, so a zero day collapsed to a few stray pixels that read as
 *    a rendering artifact rather than as "nothing happened here".
 *  - Values are **always shown**, not revealed on hover. With one busy day and
 *    six quiet ones the quiet bars are too short to compare by eye, and a
 *    hover tooltip is useless on a touchscreen and at a stand.
 *  - Non-zero bars have a real minimum height, so "1 meeting" is unmistakably
 *    taller than "0 meetings" instead of differing by two pixels.
 */

const TRACK_HEIGHT = 150

export default function WeeklyChart({ data }) {
  const max = Math.max(...data.map((d) => d.meetings), 0)

  return (
    <div className="flex-1 flex flex-col justify-end mt-auto pt-4">
      <div className="flex items-end justify-around gap-2 relative">
        {/* Scale reference — without a top tick the tallest bar has no magnitude. */}
        {max > 0 && (
          <div
            className="absolute inset-x-0 top-0 flex items-start justify-end pointer-events-none"
            style={{ height: TRACK_HEIGHT }}
          >
            <span className="font-meta-data text-[10px] text-text-faint -mt-4">max {max}</span>
          </div>
        )}

        {data.map((d, index) => {
          const isToday = d.isToday !== undefined ? d.isToday : index === data.length - 1
          const empty = d.meetings === 0
          const height = max > 0 && !empty ? Math.max((d.meetings / max) * 100, 14) : 0

          return (
            <div key={d.day} className="flex flex-col items-center gap-2 group flex-1 min-w-0 cursor-pointer">
              <span
                className={`font-meta-data text-meta-data tabular-nums transition-colors ${
                  empty
                    ? 'text-text-faint group-hover:text-text-muted'
                    : isToday
                      ? 'text-primary font-bold'
                      : 'text-text-muted group-hover:text-primary font-medium'
                }`}
              >
                {d.meetings}
              </span>

              {/* Channel: always occupies the full plot height, so an empty day
                  is a visible empty slot rather than a missing element. */}
              <div
                className="relative w-8 max-w-full rounded-t-sm bg-surface-raised/40 border-x border-t border-border/40 flex items-end overflow-hidden"
                style={{ height: TRACK_HEIGHT }}
              >
                {empty ? (
                  // Flat cap on the baseline — reads as a deliberate zero.
                  <div
                    className={`w-full h-[3px] transition-all ${
                      isToday
                        ? 'bg-primary'
                        : 'bg-border group-hover:bg-primary/70'
                    }`}
                  />
                ) : (
                  <div
                    className={`w-full rounded-t-sm transition-all duration-200 ${
                      isToday
                        ? 'bg-primary shadow-[0_0_12px_rgba(252,81,0,0.5)]'
                        : 'bg-primary/35 opacity-80 group-hover:bg-primary group-hover:opacity-100 group-hover:shadow-[0_0_10px_rgba(252,81,0,0.4)]'
                    }`}
                    style={{ height: `${height}%` }}
                  />
                )}
              </div>

              <span
                className={`font-meta-data text-meta-data truncate transition-colors ${
                  isToday ? 'text-primary font-bold' : 'text-text-muted group-hover:text-text-primary'
                }`}
              >
                {d.day}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
