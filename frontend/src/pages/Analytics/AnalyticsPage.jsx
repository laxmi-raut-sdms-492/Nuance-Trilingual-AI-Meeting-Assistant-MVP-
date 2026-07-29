import { useMemo } from 'react'
import Card from '../../components/common/Card.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import WeeklyChart from '../../components/charts/WeeklyChart.jsx'
import SpeakerPie from '../../components/charts/SpeakerPie.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

const titles = {
  insights: 'Insights',
  speakers: 'Speaker Analytics',
  activity: 'Activity'
}

const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const TYPE_COLORS = { audio: '#6366f1', video: '#22c55e', other: '#9ca3af' }

export default function AnalyticsPage({ tab }) {
  const { meetings } = useMeetings()

  const uploadsByDay = useMemo(() => {
    const counts = DAY_LABELS.map((day) => ({ day, meetings: 0 }))
    meetings.forEach((m) => {
      const idx = new Date(m.uploadedAtISO).getDay()
      counts[idx].meetings += 1
    })
    return counts
  }, [meetings])

  const typeDistribution = useMemo(() => {
    const buckets = { audio: 0, video: 0, other: 0 }
    meetings.forEach((m) => {
      if (m.fileType?.startsWith('audio')) buckets.audio += 1
      else if (m.fileType?.startsWith('video')) buckets.video += 1
      else buckets.other += 1
    })
    return Object.entries(buckets)
      .filter(([, v]) => v > 0)
      .map(([name, value]) => ({ name, value, color: TYPE_COLORS[name] }))
  }, [meetings])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{titles[tab] || 'Analytics'}</h1>
        <p className="text-sm text-gray-400 mt-1">
          Charts here are computed from meetings you've actually uploaded — speaker-level
          analytics (talk time, most active speaker) will populate once the transcription
          pipeline is connected and returns diarization data.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-2">Uploads per Week</h2>
          {meetings.length === 0 ? (
            <EmptyState title="No uploads yet" />
          ) : (
            <WeeklyChart data={uploadsByDay} />
          )}
        </Card>
        <Card>
          <h2 className="font-semibold text-gray-900 dark:text-white mb-2">File Type Distribution</h2>
          {typeDistribution.length === 0 ? (
            <EmptyState title="No data yet" />
          ) : (
            <SpeakerPie data={typeDistribution} />
          )}
        </Card>
      </div>
    </div>
  )
}
