import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ListChecks,
  CheckCircle2,
  Loader2,
  Users,
  Upload,
  ArrowUpRight,
  Clock,
  Tag
} from 'lucide-react'
import Card from '../../components/common/Card.jsx'
import Badge from '../../components/common/Badge.jsx'
import Button from '../../components/common/Button.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import StatCard from '../../components/cards/StatCard.jsx'
import WeeklyChart from '../../components/charts/WeeklyChart.jsx'
import SpeakerPie from '../../components/charts/SpeakerPie.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'
import { useMembers } from '../../context/MembersContext.jsx'
import { useUser } from '../../context/UserContext.jsx'

const statusColor = { Completed: 'green', Processing: 'yellow', Failed: 'red' }

export default function Dashboard() {
  const navigate = useNavigate()
  const { meetings } = useMeetings()
  const { members } = useMembers()
  const { profile } = useUser()

  // Everything below is derived live from real app state - nothing here is
  // hardcoded or mocked, so it stays accurate as meetings are added/removed.
  const stats = useMemo(() => {
    const total = meetings.length
    const completed = meetings.filter((m) => m.status === 'Completed').length
    const processing = meetings.filter((m) => m.status === 'Processing').length
    return { total, completed, processing }
  }, [meetings])

  const weeklyData = useMemo(() => {
    const days = []
    const today = new Date()
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today)
      d.setDate(today.getDate() - i)
      days.push({
        key: d.toDateString(),
        day: d.toLocaleDateString(undefined, { weekday: 'short' }),
        meetings: 0
      })
    }
    meetings.forEach((m) => {
      if (!m.uploadedAtISO) return
      const key = new Date(m.uploadedAtISO).toDateString()
      const bucket = days.find((d) => d.key === key)
      if (bucket) bucket.meetings += 1
    })
    return days.map(({ day, meetings }) => ({ day, meetings }))
  }, [meetings])

  const speakerPieData = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      (m.speakerStats || []).forEach((s) => {
        if (!totals[s.name]) totals[s.name] = { name: s.name, value: 0, color: s.color }
        totals[s.name].value += s.pct || 0
      })
    })
    return Object.values(totals)
  }, [meetings])

  const topKeywords = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      (m.keywords || []).forEach((k) => {
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
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {profile.name ? `Welcome back, ${profile.name.split(' ')[0]}` : 'Dashboard'}
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            {new Date().toLocaleDateString(undefined, {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            })}
          </p>
        </div>
        <Button icon={Upload} onClick={() => navigate('/upload')}>
          Create Meeting
        </Button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard label="Total Meetings" value={stats.total} icon={ListChecks} tint="primary" />
        <StatCard label="Completed" value={stats.completed} icon={CheckCircle2} tint="green" />
        <StatCard label="Processing" value={stats.processing} icon={Loader2} tint="amber" />
        <StatCard label="Team Members" value={members.length} icon={Users} tint="sky" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-4">Meeting Activity (Last 7 Days)</h2>
          {meetings.length > 0 ? (
            <WeeklyChart data={weeklyData} />
          ) : (
            <EmptyState
              title="No activity yet"
              subtitle="Upload a meeting to start seeing your weekly activity here."
            />
          )}
        </Card>

        <Card>
          <h2 className="font-semibold text-gray-900 dark:text-white mb-4">Talk-Time Share</h2>
          {speakerPieData.length > 0 ? (
            <SpeakerPie data={speakerPieData} />
          ) : (
            <EmptyState
              title="No speaker data yet"
              subtitle="Speaker breakdowns will appear here once meetings are processed."
            />
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2 !p-0 overflow-hidden">
          <div className="flex items-center justify-between px-5 pt-5 pb-3">
            <h2 className="font-semibold text-gray-900 dark:text-white">Recent Meetings</h2>
            <button
              onClick={() => navigate('/meetings')}
              className="text-xs font-semibold text-primary-600 flex items-center gap-1 hover:underline"
            >
              View all <ArrowUpRight size={13} />
            </button>
          </div>
          {recentMeetings.length === 0 ? (
            <div className="px-5 pb-5">
              <EmptyState
                title="No meetings uploaded yet"
                subtitle="Upload an audio or video recording to get started."
              />
            </div>
          ) : (
            <div className="divide-y divide-gray-50 dark:divide-gray-800">
              {recentMeetings.map((m) => (
                <button
                  key={m.id}
                  onClick={() => navigate(`/meetings/${m.id}`)}
                  className="w-full flex items-center justify-between gap-3 px-5 py-3 text-left hover:bg-gray-50/60 dark:hover:bg-gray-800/60 transition-colors"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-gray-800 dark:text-gray-100 truncate">{m.title}</p>
                    <p className="text-xs text-gray-400 flex items-center gap-1 mt-0.5">
                      <Clock size={11} /> {m.date} · {m.time}
                    </p>
                  </div>
                  <Badge color={statusColor[m.status] || 'gray'}>{m.status}</Badge>
                </button>
              ))}
            </div>
          )}
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Tag size={16} className="text-primary-500" />
            <h2 className="font-semibold text-gray-900 dark:text-white">Top Keywords</h2>
          </div>
          {topKeywords.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {topKeywords.map((k, i) => (
                <span
                  key={i}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gray-50 dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-xs font-medium text-gray-600 dark:text-gray-300"
                >
                  {k.word}
                  <span className="text-gray-400">{k.count}</span>
                </span>
              ))}
            </div>
          ) : (
            <EmptyState
              title="No keywords yet"
              subtitle="Keyword insights will show up here once meetings are processed."
            />
          )}
        </Card>
      </div>
    </div>
  )
}
