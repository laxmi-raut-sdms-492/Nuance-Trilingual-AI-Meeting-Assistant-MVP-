import { useMemo } from 'react'
import Icon from '../../components/common/Icon.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import WeeklyChart from '../../components/charts/WeeklyChart.jsx'
import SpeakerPie from '../../components/charts/SpeakerPie.jsx'
import StatCard from '../../components/cards/StatCard.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

/** Ported from the design export (insights_dashboard). */

const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const TYPE_COLORS = { audio: '#3b82f6', video: '#10b981', other: '#a1a1aa' }

import { LANGUAGE_COLORS } from '../../constants/languages.js'

const AUDIO_EXT = ['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg', 'opus', 'webm']
const VIDEO_EXT = ['mp4', 'mov', 'mkv', 'avi']

/**
 * The mime type the browser reports is not reliable: an upload made outside a
 * file picker (curl, a share sheet, some mobile browsers) arrives as
 * `application/octet-stream`, which bucketed every meeting into "other" and
 * made this chart read 100% other. Fall back to the extension, which the
 * backend preserves in `fileName`.
 */
function classifyFile(meeting) {
  const mime = meeting.fileType || ''
  if (mime.startsWith('audio')) return 'audio'
  if (mime.startsWith('video')) return 'video'

  const ext = meeting.fileName?.split('.').pop()?.toLowerCase()
  if (AUDIO_EXT.includes(ext)) return 'audio'
  if (VIDEO_EXT.includes(ext)) return 'video'
  return 'other'
}

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

  const uploadsByDay = useMemo(() => {
    const counts = DAY_LABELS.map((day) => ({ day, meetings: 0 }))
    meetings.forEach((m) => {
      if (!m.uploadedAtISO) return
      counts[new Date(m.uploadedAtISO).getDay()].meetings += 1
    })
    return counts
  }, [meetings])

  const typeDistribution = useMemo(() => {
    const buckets = { audio: 0, video: 0, other: 0 }
    meetings.forEach((m) => buckets[classifyFile(m)]++)
    return Object.entries(buckets)
      .filter(([, v]) => v > 0)
      .map(([name, value]) => ({ name, value, color: TYPE_COLORS[name] }))
  }, [meetings])

  // Both aggregations sum seconds rather than per-meeting percentages, so a
  // long meeting counts more than a short one.
  const languageMix = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      ;(m.languages || []).forEach((l) => {
        totals[l.name] = (totals[l.name] || 0) + (l.seconds || 0)
      })
    })
    return Object.entries(totals)
      .filter(([, seconds]) => seconds > 0)
      .map(([name, seconds]) => ({
        name,
        value: Math.round(seconds),
        color: LANGUAGE_COLORS[name] || '#a1a1aa',
      }))
      .sort((a, b) => b.value - a.value)
  }, [meetings])

  const talkTime = useMemo(() => {
    const totals = {}
    meetings.forEach((m) => {
      ;(m.speakerStats || []).forEach((s) => {
        if (!totals[s.name]) totals[s.name] = { name: s.name, value: 0, color: s.color }
        totals[s.name].value += s.seconds || 0
      })
    })
    return Object.values(totals)
      .filter((s) => s.value > 0)
      .map((s) => ({ ...s, value: Math.round(s.value) }))
      .sort((a, b) => b.value - a.value)
  }, [meetings])

  const totalSpokenMinutes = useMemo(
    () => Math.round(meetings.reduce((sum, m) => sum + (m.durationSeconds || 0), 0) / 60),
    [meetings]
  )

  const distinctSpeakers = talkTime.length

  return (
    <>
      <div>
        <p className="font-meta-data text-meta-data text-text-muted mb-1">Analytics</p>
        <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
          Insights
        </h2>
        <p className="font-transcript-body text-transcript-body text-text-muted mt-3 max-w-3xl">
          Computed from meetings you have actually uploaded. Talk time and language mix come from
          the diarization and per-segment language detection the backend ran on each recording —
          nothing here is estimated.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Meetings" value={meetings.length} icon="folder" tint="primary" />
        <StatCard label="Minutes Recorded" value={totalSpokenMinutes} icon="schedule" tint="primary" />
        <StatCard label="Distinct Speakers" value={distinctSpeakers} icon="group" tint="green" />
        <StatCard label="Languages Seen" value={languageMix.length} icon="translate" tint="amber" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="Uploads by Day of Week" className="lg:col-span-2 min-h-[320px]">
          {meetings.length === 0 ? (
            <EmptyState
              icon="bar_chart"
              title="No uploads yet"
              subtitle="Upload a meeting to start building activity history."
            />
          ) : (
            <WeeklyChart data={uploadsByDay} />
          )}
        </Panel>

        <Panel title="File Types">
          {typeDistribution.length === 0 ? (
            <EmptyState icon="donut_large" title="No data yet" />
          ) : (
            <SpeakerPie data={typeDistribution} />
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel
          title="Language Mix"
          subtitle="Seconds spoken in each language, detected per utterance across all meetings."
        >
          {languageMix.length === 0 ? (
            <EmptyState
              icon="translate"
              title="No language data yet"
              subtitle="This fills in once a meeting finishes transcribing."
            />
          ) : (
            <SpeakerPie data={languageMix} />
          )}
        </Panel>

        <Panel
          title="Talk Time by Speaker"
          subtitle="Seconds spoken per speaker, summed across every processed meeting."
        >
          {talkTime.length === 0 ? (
            <EmptyState
              icon="record_voice_over"
              title="No speaker data yet"
              subtitle="This fills in once a meeting finishes transcribing."
            />
          ) : (
            <SpeakerPie data={talkTime} />
          )}
        </Panel>
      </div>

      <div className="bg-surface border border-border rounded-xl p-6 flex items-start gap-3">
        <Icon name="info" className="text-text-muted shrink-0" />
        <p className="font-meta-data text-meta-data text-text-muted">
          Sentiment is not shown: nothing in the pipeline measures it. Keywords and summaries are
          produced per meeting and shown there. Only metrics the pipeline genuinely produces are
          charted here.
        </p>
      </div>
    </>
  )
}
