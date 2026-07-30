import { useMemo } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import Icon from '../../components/common/Icon.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import StatCard from '../../components/cards/StatCard.jsx'
import WeeklyChart from '../../components/charts/WeeklyChart.jsx'
import SpeakerPie from '../../components/charts/SpeakerPie.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'
import { useMembers } from '../../context/MembersContext.jsx'
import { useUser } from '../../context/UserContext.jsx'

/**
 * Ported from the design export (dashboard_populated) and
 * dashboard_zero_data/ — the same component covers both, because every panel
 * here renders a designed empty state rather than hiding.
 *
 * Everything is derived live from real meeting state. Nothing is mocked.
 */

const STATUS_STYLE = {
  Completed: { pill: 'bg-success/10 border-success/20 text-success', icon: 'check_circle', well: 'text-success', glyph: 'play_arrow' },
  Processing: { pill: 'bg-processing/10 border-processing/20 text-processing', icon: 'sync', well: 'text-processing', glyph: 'sync' },
  Failed: { pill: 'bg-error/10 border-error/20 text-error', icon: 'error', well: 'text-error', glyph: 'error' },
}

function MeetingRow({ meeting }) {
  const style = STATUS_STYLE[meeting.status] || STATUS_STYLE.Completed
  const processing = meeting.status === 'Processing'

  return (
    <Link
      to={`/meetings/${meeting.id}`}
      className={`flex items-center justify-between p-3 rounded-lg hover:bg-surface-raised cursor-pointer transition-colors border border-transparent hover:border-border group ${
        processing ? 'bg-surface-raised/50' : ''
      }`}
    >
      <div className="flex items-center gap-4 min-w-0">
        <div
          className={`w-10 h-10 rounded bg-surface-raised border border-border flex items-center justify-center shrink-0 ${style.well}`}
        >
          <Icon name={style.glyph} className={processing ? 'animate-spin text-[20px]' : ''} />
        </div>
        <div className="min-w-0">
          <p className="font-sidebar-header text-[15px] text-text-primary group-hover:text-primary transition-colors truncate">
            {meeting.title}
          </p>
          <p className="font-meta-data text-meta-data text-text-muted truncate">
            {meeting.date} · {meeting.time}
            {meeting.duration ? ` • ${meeting.duration}` : ''}
          </p>
        </div>
      </div>
      <div
        className={`px-2 py-1 rounded border font-label-sm text-[10px] uppercase tracking-wider flex items-center gap-1 shrink-0 ${style.pill}`}
      >
        <Icon name={style.icon} className={`text-[14px] ${processing ? 'animate-pulse' : ''}`} />
        {meeting.status}
      </div>
    </Link>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { meetings } = useMeetings()
  const { members } = useMembers()
  const { profile } = useUser()

  const stats = useMemo(
    () => ({
      total: meetings.length,
      completed: meetings.filter((m) => m.status === 'Completed').length,
      processing: meetings.filter((m) => m.status === 'Processing').length,
    }),
    [meetings]
  )

  const weeklyData = useMemo(() => {
    const days = []
    const today = new Date()
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today)
      d.setDate(today.getDate() - i)
      days.push({
        key: d.toDateString(),
        day: d.toLocaleDateString(undefined, { weekday: 'short' }),
        meetings: 0,
      })
    }
    meetings.forEach((m) => {
      if (!m.uploadedAtISO) return
      const key = new Date(m.uploadedAtISO).toDateString()
      const bucket = days.find((d) => d.key === key)
      if (bucket) bucket.meetings += 1
    })
    return days.map(({ day, meetings: count }) => ({ day, meetings: count }))
  }, [meetings])

  // Aggregated over seconds, not percentages. Summing each meeting's
  // percentages would weight a two-minute standup the same as a two-hour
  // review, and a speaker present in three meetings could total 300%.
  const speakerPieData = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      ;(m.speakerStats || []).forEach((s) => {
        if (!totals[s.name]) totals[s.name] = { name: s.name, value: 0, color: s.color }
        totals[s.name].value += s.seconds || 0
      })
    })
    return Object.values(totals)
      .filter((s) => s.value > 0)
      .sort((a, b) => b.value - a.value)
  }, [meetings])

  // Counts are summed across meetings, not recomputed here. The backend counts
  // each word's real occurrences in the transcript, so a tag's number is a fact;
  // inventing or rescaling it client-side would break that. Empty until at
  // least one meeting has been summarized.
  const topKeywords = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      ;(m.keywords || []).forEach((k) => {
        totals[k.word] = (totals[k.word] || 0) + (k.count || 0)
      })
    })
    return Object.entries(totals)
      .map(([word, count]) => ({ word, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)
  }, [meetings])

  const recentMeetings = useMemo(
    () =>
      [...meetings]
        .sort((a, b) => new Date(b.uploadedAtISO) - new Date(a.uploadedAtISO))
        .slice(0, 5),
    [meetings]
  )

  return (
    <>
      {/* Page header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div>
          <p className="font-meta-data text-meta-data text-text-muted mb-1">
            {new Date().toLocaleDateString(undefined, {
              weekday: 'long',
              year: 'numeric',
              month: 'short',
              day: 'numeric',
            })}
          </p>
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
            {profile.name ? `Welcome back, ${profile.name.split(' ')[0]}` : 'Dashboard'}
          </h2>
        </div>
        <button
          type="button"
          onClick={() => navigate('/upload')}
          className="bg-cta hover:bg-primary-container text-on-cta font-label-sm text-label-sm py-3 px-6 rounded-lg flex items-center gap-2 transition-all hover:scale-105 shadow-[0_0_15px_rgba(252,81,0,0.3)]"
        >
          <Icon name="videocam" />
          Create Meeting
        </button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Meetings" value={stats.total} icon="folder" tint="primary" />
        <StatCard label="Completed" value={stats.completed} icon="check_circle" tint="green" />
        <StatCard label="Processing" value={stats.processing} icon="sync" tint="amber" />
        <StatCard label="Team Members" value={members.length} icon="groups" tint="primary" />
      </div>

      {/* Bento row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-auto lg:h-[300px]">
        <div className="bg-surface border border-border rounded-xl p-6 lg:col-span-2 flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
              Meeting Activity (Last 7 Days)
            </h3>
          </div>
          {meetings.length > 0 ? (
            <WeeklyChart data={weeklyData} />
          ) : (
            <EmptyState
              icon="bar_chart"
              title="No activity yet"
              subtitle="Upload a meeting to start seeing your weekly activity here."
            />
          )}
        </div>

        <div className="bg-surface border border-border rounded-xl p-6 flex flex-col items-center justify-center relative overflow-hidden">
          <h3 className="font-sidebar-header text-sidebar-header text-text-primary absolute top-6 left-6">
            Talk-Time Share
          </h3>
          {speakerPieData.length > 0 ? (
            <div className="mt-8">
              <SpeakerPie data={speakerPieData} />
            </div>
          ) : (
            <EmptyState
              icon="donut_large"
              title="No speaker data yet"
              subtitle="Breakdowns appear once a meeting finishes processing."
            />
          )}
        </div>
      </div>

      {/* Bento row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-surface border border-border rounded-xl p-6 lg:col-span-2">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
              Recent Meetings
            </h3>
            <Link
              to="/meetings"
              className="font-label-sm text-label-sm text-primary hover:text-primary-container transition-colors"
            >
              View all
            </Link>
          </div>
          {recentMeetings.length === 0 ? (
            <EmptyState
              icon="event_note"
              title="No meetings uploaded yet"
              subtitle="Upload an audio or video recording to get started."
            />
          ) : (
            <div className="flex flex-col gap-2">
              {recentMeetings.map((m) => (
                <MeetingRow key={m.id} meeting={m} />
              ))}
            </div>
          )}
        </div>

        <div className="bg-surface border border-border rounded-xl p-6">
          <div className="flex items-center gap-2 mb-6">
            <Icon name="sell" className="text-primary" />
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
              Top Keywords
            </h3>
          </div>
          {topKeywords.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {topKeywords.map((k) => (
                <span
                  key={k.word}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-surface-raised border border-border font-meta-data text-meta-data text-text-muted"
                >
                  {k.word}
                  <span className="text-text-faint">{k.count}</span>
                </span>
              ))}
            </div>
          ) : (
            <EmptyState
              icon="sell"
              title="No keywords yet"
              subtitle="Keywords appear once a meeting has been summarized. They are counted from the transcript, so an empty panel means nothing has been processed yet."
            />
          )}
        </div>
      </div>
    </>
  )
}
