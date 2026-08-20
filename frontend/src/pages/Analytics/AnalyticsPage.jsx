import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import Icon from '../../components/common/Icon.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import WeeklyChart from '../../components/charts/WeeklyChart.jsx'
import StatCard from '../../components/cards/StatCard.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function Panel({ title, subtitle, children, className = '' }) {
  return (
    <div className={`bg-surface border border-border rounded-xl p-6 flex flex-col ${className}`}>
      <h3 className="font-sidebar-header text-sidebar-header text-text-primary">{title}</h3>
      {subtitle && (
        <p className="font-meta-data text-meta-data text-text-muted mt-1 mb-4">{subtitle}</p>
      )}
      <div className={`flex-1 flex flex-col justify-center ${subtitle ? '' : 'mt-4'}`}>
        {children}
      </div>
    </div>
  )
}

export default function AnalyticsPage() {
  const { meetings } = useMeetings()
  const [selectedMeetingId, setSelectedMeetingId] = useState('')
  const [openMeetingIds, setOpenMeetingIds] = useState([])
  const [dateRange, setDateRange] = useState('7days')
  const [customDate, setCustomDate] = useState('')
  const [activeInsightModal, setActiveInsightModal] = useState(null)

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

  const uploadsByDay = useMemo(() => {
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
      let dateObj = null
      if (m.uploadedAtISO) dateObj = new Date(m.uploadedAtISO)
      else if (m.uploadedAt) dateObj = new Date(m.uploadedAt)
      else if (m.date && m.date !== '—') dateObj = new Date(m.date)
      if (!dateObj || isNaN(dateObj.getTime())) dateObj = new Date()

      const key = dateObj.toDateString()
      const bucket = days.find((d) => d.key === key)
      if (bucket) bucket.meetings += 1
    })
    return days.map(({ day, isToday, meetings: count }) => ({ day, isToday, meetings: count }))
  }, [filteredMeetingsForChart])

  const allMeetingsList = useMemo(() => {
    return meetings.map((m) => {
      const rawTasks = []

      // 1. Direct actionItems
      if (Array.isArray(m.actionItems)) {
        m.actionItems.forEach((item, idx) => {
          if (!item) return
          const text = typeof item === 'string' ? item : item.text || item.title || item.task
          if (text) {
            rawTasks.push({
              id: `${m.id}-ai-${idx}`,
              text,
              assignee: item.assignee || item.owner || 'Unassigned',
              dueDate: item.due || item.deadline || 'Not set',
              completed: item.status?.toLowerCase() === 'completed' || item.completed || false,
            })
          }
        })
      }

      // 2. insights.pending
      if (Array.isArray(m.insights?.pending)) {
        m.insights.pending.forEach((item, idx) => {
          if (!item) return
          const text = typeof item === 'string' ? item : item.text || item.title || item.task
          if (text && !rawTasks.some((t) => t.text === text)) {
            rawTasks.push({
              id: `${m.id}-pen-${idx}`,
              text,
              assignee: item.owner || item.assignee || 'Unassigned',
              dueDate: item.due || item.deadline || 'Not set',
              completed: false,
            })
          }
        })
      }

      // 3. summaryData?.actionItems
      if (Array.isArray(m.summaryData?.actionItems)) {
        m.summaryData.actionItems.forEach((item, idx) => {
          if (!item) return
          const text = typeof item === 'string' ? item : item.text || item.title || item.task
          if (text && !rawTasks.some((t) => t.text === text)) {
            rawTasks.push({
              id: `${m.id}-sum-${idx}`,
              text,
              assignee: 'Unassigned',
              dueDate: 'Not set',
              completed: false,
            })
          }
        })
      }

      // 4. insights.commitments
      if (Array.isArray(m.insights?.commitments)) {
        m.insights.commitments.forEach((item, idx) => {
          if (!item) return
          const text = typeof item === 'string' ? item : item.text || item.commitment || item.title
          if (text && !rawTasks.some((t) => t.text === text)) {
            rawTasks.push({
              id: `${m.id}-com-${idx}`,
              text,
              assignee: item.speaker || item.assignee || 'Unassigned',
              dueDate: 'Not set',
              completed: false,
            })
          }
        })
      }

      return {
        id: m.id,
        title: m.title || 'Untitled Meeting',
        date: m.date || (m.uploadedAtISO ? new Date(m.uploadedAtISO).toLocaleDateString() : ''),
        time: m.time,
        tasks: rawTasks,
      }
    })
  }, [meetings])

  const pendingTasksByMeeting = useMemo(() => {
    return allMeetingsList.filter((m) => m.tasks.length > 0)
  }, [allMeetingsList])

  const totalPendingCount = useMemo(() => {
    return allMeetingsList.reduce((sum, m) => sum + m.tasks.length, 0)
  }, [allMeetingsList])

  const totalSpokenMinutes = useMemo(
    () => Math.round(meetings.reduce((sum, m) => sum + (m.durationSeconds || 0), 0) / 60),
    [meetings]
  )

  const avgMeetingDuration = useMemo(() => {
    if (meetings.length === 0) return '0 min'
    const totalSec = meetings.reduce((sum, m) => sum + (m.durationSeconds || 0), 0)
    const avgSec = Math.round(totalSec / meetings.length)
    const mins = Math.floor(avgSec / 60)
    const secs = avgSec % 60
    if (mins === 0) return `${secs} sec`
    if (secs === 0) return `${mins} min`
    return `${mins}m ${secs}s`
  }, [meetings])

  const meetingsSortedByDuration = useMemo(() => {
    return [...meetings].sort((a, b) => (b.durationSeconds || 0) - (a.durationSeconds || 0))
  }, [meetings])

  const activeFilter = selectedMeetingId || 'all'
  const displayedMeetings = useMemo(() => {
    if (activeFilter === 'all') {
      const withTasks = allMeetingsList.filter((m) => m.tasks.length > 0)
      return withTasks.length > 0 ? withTasks : allMeetingsList
    }
    return allMeetingsList.filter((m) => m.id === activeFilter)
  }, [activeFilter, allMeetingsList])

  return (
    <div className="flex flex-col gap-6">
      <div>
        <p className="font-meta-data text-meta-data text-text-muted mb-1">Analytics</p>
        <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
          Insights
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          label="Total Meetings"
          value={meetings.length}
          icon="folder"
          tint="primary"
        />
        <StatCard
          label="Minutes Recorded"
          value={totalSpokenMinutes}
          icon="schedule"
          tint="primary"
        />
        <StatCard
          label="Avg Duration"
          value={avgMeetingDuration}
          icon="timer"
          tint="primary"
        />
        <StatCard
          label="Total Pending Tasks"
          value={totalPendingCount}
          icon="task_alt"
          tint="amber"
        />
      </div>

      {/* Side-by-Side Grid: Uploads Chart + Pending Tasks Dropdown Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full items-start">
        {/* Uploads Chart Panel with Calendar Date Dropdown */}
        <div className="bg-surface border border-border rounded-xl p-5 flex flex-col w-full min-h-[300px] h-full">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3 pb-2 border-b border-border/40">
            <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
              Uploads by Day of Week
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
          {meetings.length === 0 ? (
            <EmptyState
              icon="bar_chart"
              title="No uploads yet"
              subtitle="Upload a meeting to start building activity history."
            />
          ) : (
            <WeeklyChart data={uploadsByDay} />
          )}
        </div>

        {/* Meeting-Wise Pending Tasks Dropdown Panel */}
        <div id="pending-tasks-panel" className="bg-surface border border-border rounded-xl p-5 flex flex-col w-full gap-4 min-h-[300px] h-full scroll-mt-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
            <div>
              <h3 className="font-sidebar-header text-sidebar-header text-text-primary">
                Pending Tasks
              </h3>
              <p className="font-meta-data text-meta-data text-text-muted text-xs">
                Select a meeting to filter its action items
              </p>
            </div>

            {/* Meeting Filter Dropdown */}
            <div className="flex items-center gap-2 bg-surface-raised border border-border px-3 py-1.5 rounded-lg shrink-0">
              <Icon name="filter_list" className="text-text-muted" size={16} />
              <select
                value={activeFilter}
                onChange={(e) => setSelectedMeetingId(e.target.value === 'all' ? '' : e.target.value)}
                className="bg-transparent font-meta-data text-meta-data text-text-primary focus:outline-none text-xs cursor-pointer max-w-[200px] truncate"
              >
                <option value="all">All Meetings ({allMeetingsList.length})</option>
                {allMeetingsList.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.title} ({m.tasks.length} {m.tasks.length === 1 ? 'task' : 'tasks'})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Collapsible Accordion per Meeting */}
          {displayedMeetings.length === 0 ? (
            <EmptyState
              icon="task_alt"
              title="No meetings uploaded yet"
              subtitle="Uploaded meetings will display here with their action items."
            />
          ) : (
            <div className="flex flex-col gap-3 max-h-[420px] overflow-y-auto pr-1">
              {displayedMeetings.map((meeting) => {
                const isOpen = openMeetingIds.includes(meeting.id) || activeFilter !== 'all'
                return (
                  <div
                    key={meeting.id}
                    className="border border-border rounded-lg bg-surface-raised/50 overflow-hidden transition-all"
                  >
                    {/* Accordion Header */}
                    <button
                      type="button"
                      onClick={() => {
                        setOpenMeetingIds((prev) =>
                          prev.includes(meeting.id)
                            ? prev.filter((id) => id !== meeting.id)
                            : [...prev, meeting.id]
                        )
                      }}
                      className="w-full flex items-center justify-between p-3.5 bg-surface-raised hover:bg-surface transition-colors text-left"
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <Icon
                          name={isOpen ? 'expand_more' : 'chevron_right'}
                          size={18}
                          className="text-text-muted shrink-0 transition-transform"
                        />
                        <span className="font-sidebar-header text-xs font-bold text-text-primary truncate">
                          {meeting.title}
                        </span>
                        <span className="font-meta-data text-[10px] text-text-muted shrink-0">
                          ({meeting.date || '—'})
                        </span>
                      </div>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide bg-amber-500/10 text-amber-500 border border-amber-500/20 shrink-0">
                        {meeting.tasks.length} {meeting.tasks.length === 1 ? 'Task' : 'Tasks'}
                      </span>
                    </button>

                    {/* Accordion Content */}
                    {isOpen && (
                      <div className="p-3.5 border-t border-border/40 bg-surface flex flex-col gap-2.5">
                        {meeting.tasks.length === 0 ? (
                          <p className="font-meta-data text-xs text-text-muted italic py-1">
                            No action items generated for this meeting yet.
                          </p>
                        ) : (
                          meeting.tasks.map((task) => (
                            <div
                              key={task.id}
                              className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2.5 rounded-lg border border-border/60 bg-surface-raised text-xs"
                            >
                              <div className="flex items-center gap-2.5 min-w-0">
                                <Icon name="check_box_outline_blank" size={16} className="text-amber-500 shrink-0" />
                                <span className="font-medium text-text-primary truncate">{task.text}</span>
                              </div>
                              <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto font-meta-data text-[11px] text-text-muted">
                                <span className="bg-surface px-2 py-0.5 rounded border border-border text-text-muted">
                                  {task.assignee}
                                </span>
                                <span className="bg-surface px-2 py-0.5 rounded border border-border text-text-faint">
                                  {task.dueDate}
                                </span>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Insight Breakdown Modal */}
      {activeInsightModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
          <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-xl shadow-2xl flex flex-col gap-4 relative max-h-[85vh]">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Icon
                  name={
                    activeInsightModal === 'totalMeetings'
                      ? 'folder'
                      : activeInsightModal === 'minutesRecorded'
                      ? 'schedule'
                      : activeInsightModal === 'avgDuration'
                      ? 'timer'
                      : activeInsightModal === 'meetingTasks'
                      ? 'event_available'
                      : 'task_alt'
                  }
                  className="text-primary"
                  size={22}
                />
                <h3 className="font-sidebar-header text-sidebar-header text-text-primary text-base font-bold">
                  {activeInsightModal === 'totalMeetings' && `Total Meetings (${meetings.length})`}
                  {activeInsightModal === 'minutesRecorded' && `Minutes Recorded Breakdown (${totalSpokenMinutes} mins)`}
                  {activeInsightModal === 'avgDuration' && `Average Duration Breakdown (${avgMeetingDuration} avg)`}
                  {activeInsightModal === 'meetingTasks' && `Meetings with Tasks (${pendingTasksByMeeting.length} Meetings)`}
                  {activeInsightModal === 'pendingTasks' && `All Pending Tasks Breakdown (${totalPendingCount} Pending)`}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setActiveInsightModal(null)}
                className="p-1 rounded text-text-muted hover:text-text-primary transition-colors focus:outline-none"
              >
                <Icon name="close" size={20} />
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex flex-col gap-3 overflow-y-auto pr-1 max-h-[60vh]">
              {/* 1. Total Meetings Modal Content */}
              {activeInsightModal === 'totalMeetings' && (
                meetings.length === 0 ? (
                  <EmptyState icon="folder" title="No meetings found" subtitle="Upload a meeting recording to see it listed here." />
                ) : (
                  meetings.map((m) => {
                    const durSec = Math.round(m.durationSeconds || 0)
                    const mins = Math.floor(durSec / 60)
                    const secs = durSec % 60
                    const durLabel = mins > 0 ? (secs > 0 ? `${mins}m ${secs}s` : `${mins}m`) : `${secs}s`
                    return (
                      <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border border-border/60 bg-surface-raised">
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="w-8 h-8 rounded-lg bg-surface border border-border flex items-center justify-center text-primary shrink-0">
                            <Icon name="folder" size={16} />
                          </div>
                          <div className="min-w-0">
                            <h4 className="font-sidebar-header text-xs font-bold text-text-primary truncate">{m.title}</h4>
                            <p className="font-meta-data text-[11px] text-text-muted">{m.date || '—'} · {m.time || '—'}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          <span className="font-meta-data text-xs text-text-muted bg-surface px-2 py-1 rounded border border-border">{durLabel}</span>
                          <Link to={`/meetings/${m.id}`} className="text-xs text-primary font-medium hover:underline flex items-center gap-1">
                            View <Icon name="arrow_forward" size={12} />
                          </Link>
                        </div>
                      </div>
                    )
                  })
                )
              )}

              {/* 2. Minutes Recorded Breakdown */}
              {activeInsightModal === 'minutesRecorded' && (
                meetingsSortedByDuration.length === 0 ? (
                  <EmptyState icon="schedule" title="No recorded minutes" subtitle="Upload a meeting to calculate recorded minutes." />
                ) : (
                  meetingsSortedByDuration.map((m) => {
                    const durSec = Math.round(m.durationSeconds || 0)
                    const mins = Math.floor(durSec / 60)
                    const secs = durSec % 60
                    const durLabel = mins > 0 ? (secs > 0 ? `${mins}m ${secs}s` : `${mins}m`) : `${secs}s`
                    return (
                      <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border border-border/60 bg-surface-raised">
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shrink-0">
                            <Icon name="schedule" size={16} />
                          </div>
                          <div className="min-w-0">
                            <h4 className="font-sidebar-header text-xs font-bold text-text-primary truncate">{m.title}</h4>
                            <p className="font-meta-data text-[11px] text-text-muted">{m.date || '—'}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          <span className="font-bold text-xs text-primary bg-primary/10 px-2.5 py-1 rounded-full border border-primary/20">{durLabel}</span>
                          <Link to={`/meetings/${m.id}`} className="text-xs text-primary font-medium hover:underline flex items-center gap-1">
                            View <Icon name="arrow_forward" size={12} />
                          </Link>
                        </div>
                      </div>
                    )
                  })
                )
              )}

              {/* 3. Average Duration Breakdown */}
              {activeInsightModal === 'avgDuration' && (
                meetingsSortedByDuration.length === 0 ? (
                  <EmptyState icon="timer" title="No meetings to average" subtitle="Upload meetings to compute average duration." />
                ) : (
                  <>
                    <div className="p-3 bg-primary/10 border border-primary/20 rounded-lg flex justify-between items-center mb-1">
                      <span className="font-sidebar-header text-xs font-bold text-primary">Average Meeting Duration</span>
                      <span className="font-bold text-sm text-primary">{avgMeetingDuration}</span>
                    </div>
                    {meetingsSortedByDuration.map((m) => {
                      const durSec = Math.round(m.durationSeconds || 0)
                      const mins = Math.floor(durSec / 60)
                      const secs = durSec % 60
                      const durLabel = mins > 0 ? (secs > 0 ? `${mins}m ${secs}s` : `${mins}m`) : `${secs}s`
                      return (
                        <div key={m.id} className="flex items-center justify-between p-3 rounded-lg border border-border/60 bg-surface-raised gap-2">
                          <div className="flex items-center gap-3 min-w-0">
                            <div className="w-8 h-8 rounded-lg bg-surface border border-border flex items-center justify-center text-text-muted shrink-0">
                              <Icon name="timer" size={16} />
                            </div>
                            <div className="min-w-0">
                              <div className="flex items-center gap-1.5 text-xs truncate">
                                <span className="text-text-muted font-medium shrink-0">Meeting Name:</span>
                                <span className="font-bold text-text-primary truncate">{m.title}</span>
                              </div>
                              <p className="font-meta-data text-[11px] text-text-muted mt-0.5">{m.date || '—'}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2.5 shrink-0">
                            <div className="flex items-center gap-1 text-xs bg-surface px-2.5 py-1 rounded border border-border">
                              <span className="text-text-muted font-medium">Duration:</span>
                              <span className="font-bold text-text-primary">{durLabel}</span>
                            </div>
                            <Link to={`/meetings/${m.id}`} className="text-xs text-primary font-medium hover:underline flex items-center gap-1">
                              View <Icon name="arrow_forward" size={12} />
                            </Link>
                          </div>
                        </div>
                      )
                    })}
                  </>
                )
              )}



              {/* 5. Total Pending Tasks Modal */}
              {activeInsightModal === 'pendingTasks' && (
                pendingTasksByMeeting.length === 0 ? (
                  <EmptyState icon="task_alt" title="No pending tasks" subtitle="Great job! All meeting tasks are complete." />
                ) : (
                  pendingTasksByMeeting.map((m) => {
                    const uncompletedTasks = m.tasks?.filter((t) => !t.completed) || []
                    if (uncompletedTasks.length === 0) return null
                    return (
                      <div key={m.id} className="p-3 rounded-lg border border-border/60 bg-surface-raised flex flex-col gap-2">
                        <div className="flex items-center justify-between border-b border-border/40 pb-1.5">
                          <h4 className="font-sidebar-header text-xs font-bold text-text-primary truncate">{m.title}</h4>
                          <Link to={`/meetings/${m.id}`} className="text-xs text-primary font-medium hover:underline flex items-center gap-1">
                            View Meeting <Icon name="arrow_forward" size={12} />
                          </Link>
                        </div>
                        <div className="flex flex-col gap-1.5 pl-1">
                          {uncompletedTasks.map((t) => (
                            <div key={t.id} className="flex items-center justify-between p-2 rounded bg-surface border border-border/40 text-xs">
                              <div className="flex items-center gap-2 min-w-0">
                                <Icon name="radio_button_unchecked" size={14} className="text-amber-500 shrink-0" />
                                <span className="text-text-primary font-medium truncate">{t.text}</span>
                              </div>
                              <span className="font-meta-data text-[10px] bg-surface-raised px-2 py-0.5 rounded text-text-muted border border-border shrink-0 ml-2">
                                {t.assignee || 'Unassigned'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })
                )
              )}
            </div>

            {/* Modal Footer */}
            <div className="pt-2 flex justify-end border-t border-border">
              <button
                type="button"
                onClick={() => setActiveInsightModal(null)}
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
