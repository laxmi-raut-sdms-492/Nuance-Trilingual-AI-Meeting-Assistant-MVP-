import { useMemo, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import Icon from '../../components/common/Icon.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import StatCard from '../../components/cards/StatCard.jsx'
import WeeklyChart from '../../components/charts/WeeklyChart.jsx'
import SpeakerPie from '../../components/charts/SpeakerPie.jsx'
import CategoryPie from '../../components/charts/CategoryPie.jsx'
import DepartmentChart from '../../components/charts/DepartmentChart.jsx'
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
  const isInternal = meeting.meetingType === 'internal' || !meeting.meetingType

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
          <div className="flex items-center gap-2 flex-wrap mt-0.5">
            <span className="font-meta-data text-meta-data text-text-muted truncate">
              {meeting.date} · {meeting.time}
              {meeting.duration ? ` • ${meeting.duration}` : ''}
            </span>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${isInternal ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : 'bg-purple-500/10 text-purple-400 border-purple-500/20'}`}>
              {isInternal ? ' Internal' : 'Client'}
            </span>
            {meeting.department && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-surface-raised text-text-muted border border-border">
                {meeting.department}
              </span>
            )}
            {meeting.projectName && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-surface-raised text-text-faint border border-border">
                {meeting.projectName}
              </span>
            )}
          </div>
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
  const [dateRange, setDateRange] = useState('7days')
  const [customDate, setCustomDate] = useState('')
  const [activeStatusFilter, setActiveStatusFilter] = useState('all')
  const [showMembersModal, setShowMembersModal] = useState(false)

  const stats = useMemo(
    () => ({
      total: meetings.length,
      completed: meetings.filter((m) => m.status === 'Completed').length,
      processing: meetings.filter((m) => m.status === 'Processing').length,
    }),
    [meetings]
  )

  const filteredMeetingsForChart = useMemo(() => {
    const now = new Date()
    return meetings.filter((m) => {
      if (!m.uploadedAtISO) return true
      const mDate = new Date(m.uploadedAtISO)
      if (dateRange === '7days') {
        const diffDays = (now - mDate) / (1000 * 60 * 60 * 24)
        return diffDays <= 7
      }
      if (dateRange === '30days') {
        const diffDays = (now - mDate) / (1000 * 60 * 60 * 24)
        return diffDays <= 30
      }
      if (dateRange === 'thisMonth') {
        return mDate.getMonth() === now.getMonth() && mDate.getFullYear() === now.getFullYear()
      }
      if (dateRange === 'custom' && customDate) {
        const targetStr = new Date(customDate).toDateString()
        return mDate.toDateString() === targetStr
      }
      return true
    })
  }, [meetings, dateRange, customDate])

  const weeklyData = useMemo(() => {
    const days = []
    const today = new Date()
    const todayStr = today.toDateString()
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today)
      d.setDate(today.getDate() - i)
      const key = d.toDateString()
      days.push({
        key,
        day: d.toLocaleDateString(undefined, { weekday: 'short' }),
        isToday: key === todayStr,
        meetings: 0,
      })
    }
    filteredMeetingsForChart.forEach((m) => {
      if (!m.uploadedAtISO) return
      const key = new Date(m.uploadedAtISO).toDateString()
      const bucket = days.find((d) => d.key === key)
      if (bucket) bucket.meetings += 1
    })
    return days.map(({ day, isToday, meetings: count }) => ({ day, isToday, meetings: count }))
  }, [filteredMeetingsForChart])

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


  const filteredRecentMeetings = useMemo(() => {
    let list = meetings
    if (activeStatusFilter !== 'all') {
      list = meetings.filter((m) => m.status?.toLowerCase() === activeStatusFilter.toLowerCase())
    }
    return [...list]
      .sort((a, b) => new Date(b.uploadedAtISO) - new Date(a.uploadedAtISO))
      .slice(0, 5)
  }, [meetings, activeStatusFilter])

  const categoryStats = useMemo(() => {
    let internal = 0
    let client = 0
    const depts = {}
    meetings.forEach((m) => {
      if (m.meetingType === 'client') client++
      else internal++
      const dept = m.department || 'AI Team'
      depts[dept] = (depts[dept] || 0) + 1
    })
    return { internal, client, depts }
  }, [meetings])

  return (
    <div className="flex flex-col gap-5">
      {/* Page header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <p className="font-meta-data text-meta-data text-text-muted mb-0.5">
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
          className="bg-cta hover:bg-primary-container text-on-cta font-label-sm text-label-sm py-2.5 px-5 rounded-lg flex items-center gap-2 transition-all hover:scale-105 shadow-[0_0_15px_rgba(252,81,0,0.3)] shrink-0"
        >
          <Icon name="videocam" />
          Create Meeting
        </button>
      </div>

      {/* Quick Stats Grid - Full Width Top Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 w-full">
        <StatCard
          label="Total Meetings"
          value={stats.total}
          icon="folder"
          tint="primary"
        />
        <StatCard
          label="Completed"
          value={stats.completed}
          icon="check_circle"
          tint="green"
        />
        <StatCard
          label="Processing"
          value={stats.processing}
          icon="sync"
          tint="amber"
        />
        <StatCard
          label="Team Members"
          value={members.length}
          icon="groups"
          tint="primary"
        />
      </div>

      {/* Interactive Category & Department Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full">
        {/* Category Distribution Donut Chart */}
        <div className="bg-surface border border-border rounded-xl p-5 flex flex-col justify-between min-h-[220px]">
          <div className="flex items-center justify-between pb-3 border-b border-border/40">
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary flex items-center gap-2">
              <Icon name="pie_chart" size={18} className="text-primary" />
              <span>Meeting Categories</span>
            </h3>
            <span className="font-meta-data text-[11px] text-text-muted bg-surface-raised px-2 py-0.5 rounded border border-border">
              Internal vs Client
            </span>
          </div>
          <div className="py-2 flex-1">
            <CategoryPie meetings={meetings} />
          </div>
        </div>

        {/* Department Volume Horizontal Bar Chart */}
        <div className="bg-surface border border-border rounded-xl p-5 flex flex-col justify-between min-h-[220px]">
          <div className="flex items-center justify-between pb-3 border-b border-border/40">
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary flex items-center gap-2">
              <Icon name="bar_chart" size={18} className="text-primary" />
              <span>Department Breakdown</span>
            </h3>
            <span className="font-meta-data text-[11px] text-text-muted bg-surface-raised px-2 py-0.5 rounded border border-border">
              Volume by Dept
            </span>
          </div>
          <div className="py-2 flex-1">
            <DepartmentChart meetings={meetings} />
          </div>
        </div>
      </div>

      {/* Side-by-Side Grid Layout: Meeting Activity Chart (Left) + Recent Meetings (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start w-full">
        {/* Meeting Activity Chart */}
        <div className="bg-surface border border-border rounded-xl p-5 flex flex-col min-h-[300px]">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2 pb-2 border-b border-border/40">
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
              Meeting Activity
            </h3>
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-1.5 bg-surface-raised border border-border px-2.5 py-1 rounded-lg">
                <Icon name="filter_alt" className="text-text-muted" size={14} />
                <select
                  value={customDate ? 'custom' : dateRange}
                  onChange={(e) => {
                    setDateRange(e.target.value)
                    setCustomDate('')
                  }}
                  className="bg-transparent font-meta-data text-meta-data text-text-primary focus:outline-none text-xs cursor-pointer"
                >
                  <option value="7days">Last 7 Days</option>
                  <option value="30days">Last 30 Days</option>
                  <option value="thisMonth">This Month</option>
                  <option value="allTime">All Time</option>
                </select>
              </div>

              <div className="relative flex items-center">
                <input
                  type="date"
                  max={new Date().toISOString().split('T')[0]}
                  value={customDate}
                  onChange={(e) => {
                    setCustomDate(e.target.value)
                    if (e.target.value) setDateRange('custom')
                  }}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
                />
                <div className="input-base px-2.5 py-1 rounded-lg border border-border font-meta-data text-meta-data bg-surface-raised text-text-primary flex items-center gap-2 text-xs pointer-events-none">
                  <span className={customDate ? 'text-text-primary font-medium' : 'text-text-muted'}>
                    {customDate ? (
                      (() => {
                        const parts = customDate.split('-')
                        return parts.length === 3 ? `${parts[2]}/${parts[1]}/${parts[0]}` : customDate
                      })()
                    ) : (
                      'dd/mm/yyyy'
                    )}
                  </span>
                  <Icon name="calendar_today" size={14} className="text-text-muted" />
                </div>
              </div>
            </div>
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

        {/* Recent Meetings */}
        <div className="bg-surface border border-border rounded-xl p-5 flex flex-col min-h-[300px]">
          <div className="flex justify-between items-center mb-3 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
                {activeStatusFilter === 'all' ? 'Recent Meetings' : `Recent Meetings (${activeStatusFilter})`}
              </h3>
              {activeStatusFilter !== 'all' && (
                <button
                  type="button"
                  onClick={() => setActiveStatusFilter('all')}
                  className="text-xs text-primary hover:underline font-meta-data font-semibold"
                >
                  (Show all)
                </button>
              )}
            </div>
            <Link
              to={activeStatusFilter === 'all' ? '/meetings' : `/meetings?status=${activeStatusFilter}`}
              className="font-label-sm text-label-sm text-primary hover:text-primary-container transition-colors"
            >
              View all →
            </Link>
          </div>
          {filteredRecentMeetings.length === 0 ? (
            <EmptyState
              icon="event_note"
              title={activeStatusFilter === 'all' ? 'No meetings uploaded yet' : `No ${activeStatusFilter} meetings found`}
              subtitle={activeStatusFilter === 'all' ? 'Upload an audio or video recording to get started.' : 'Try selecting another tab or clear filter.'}
            />
          ) : (
            <div className="flex flex-col gap-2">
              {filteredRecentMeetings.map((m) => (
                <MeetingRow key={m.id} meeting={m} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Team Members Modal */}
      {showMembersModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
          <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-md shadow-2xl flex flex-col gap-4 relative">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Icon name="groups" className="text-primary" size={22} />
                <h3 className="font-sidebar-header text-sidebar-header text-text-primary text-base font-bold">
                  Team Members
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setShowMembersModal(false)}
                className="p-1 rounded text-text-muted hover:text-text-primary transition-colors focus:outline-none"
              >
                <Icon name="close" size={20} />
              </button>
            </div>

            <div className="flex flex-col gap-2.5 max-h-[300px] overflow-y-auto pr-1">
              {members.map((m) => {
                const initials = m.name ? m.name.slice(0, 2).toUpperCase() : 'AN'
                return (
                  <div
                    key={m.id}
                    className="flex items-center gap-3 p-3 rounded-lg border border-border/60 bg-surface-raised"
                  >
                    <div className="w-9 h-9 rounded-full bg-primary text-white font-bold text-xs flex items-center justify-center shrink-0">
                      {initials}
                    </div>
                    <div className="min-w-0 flex-1">
                      <h4 className="font-sidebar-header text-xs font-bold text-text-primary truncate">
                        {m.name}
                      </h4>
                      <p className="font-meta-data text-xs text-text-muted truncate">
                        {m.email}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="pt-2 flex justify-end">
              <button
                type="button"
                onClick={() => setShowMembersModal(false)}
                className="px-4 py-1.5 text-xs bg-primary text-white font-medium rounded-lg hover:bg-primary/90 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
