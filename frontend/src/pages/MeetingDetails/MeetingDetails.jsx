import { useParams, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import {
  Calendar,
  Clock,
  FileAudio,
  Share2,
  Download,
  Search,
  Sparkles,
  CheckCircle2,
  Lightbulb,
  Info
} from 'lucide-react'
import Card from '../../components/common/Card.jsx'
import Badge from '../../components/common/Badge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import AudioPlayer from '../../components/common/AudioPlayer.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

const tabs = ['Transcript', 'Summary', 'Insights']
const statusColor = { Completed: 'green', Processing: 'yellow', Failed: 'red' }

export default function MeetingDetails() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { getById } = useMeetings()
  const meeting = getById(id)
  const [activeTab, setActiveTab] = useState('Transcript')
  const [search, setSearch] = useState('')

  if (!meeting) {
    return (
      <EmptyState
        title="Meeting not found"
        subtitle="It may have been deleted, or the link is incorrect."
      />
    )
  }

  const hasTranscript = meeting.transcript && meeting.transcript.length > 0
  const filteredTranscript = hasTranscript
    ? meeting.transcript.filter((t) => t.text.toLowerCase().includes(search.toLowerCase()))
    : []

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{meeting.title}</h1>
          <div className="flex flex-wrap items-center gap-2 mt-3">
            <MetaChip icon={Calendar}>{meeting.date}</MetaChip>
            <MetaChip icon={Clock}>{meeting.time}</MetaChip>
            <MetaChip icon={FileAudio}>{meeting.fileSizeLabel}</MetaChip>
            <Badge color={statusColor[meeting.status] || 'gray'}>{meeting.status}</Badge>
          </div>
          {meeting.agenda && (
            <p className="text-sm text-gray-500 mt-3 max-w-2xl leading-relaxed">
              <span className="font-semibold text-gray-700">Agenda: </span>
              {meeting.agenda}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button className="flex items-center gap-2 px-4 py-2 rounded-xl border border-gray-200 text-sm font-semibold text-gray-700 hover:bg-gray-50">
            <Share2 size={15} /> Share
          </button>
          <button
            disabled={!hasTranscript}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-primary-600 text-white text-sm font-semibold hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download size={15} /> Download
          </button>
        </div>
      </div>

      <AudioPlayer meetingId={meeting.id} fileName={meeting.fileName} />

      {meeting.status === 'Processing' && (
        <Card className="bg-amber-50/60 border-amber-100">
          <p className="text-sm text-amber-700">
            This file has been saved, but transcription hasn't run yet — connect the FSD backend
            (Replicate WhisperX + OpenRouter) to generate a real transcript, summary, and action
            items for it.
          </p>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left / Main column */}
        <div className="lg:col-span-2 space-y-5">
          <Card className="!p-0 overflow-hidden">
            <div className="flex border-b border-gray-100 px-2">
              {tabs.map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${
                    activeTab === t
                      ? 'border-primary-600 text-primary-600'
                      : 'border-transparent text-gray-400 hover:text-gray-600'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>

            <div className="p-5">
              {activeTab === 'Transcript' &&
                (hasTranscript ? (
                  <div className="space-y-4 max-h-[420px] overflow-y-auto pr-1">
                    <div className="relative max-w-xs mb-2">
                      <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                      <input
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search transcript..."
                        className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-gray-200 text-xs focus:outline-none focus:ring-2 focus:ring-primary-200"
                      />
                    </div>
                    {filteredTranscript.length === 0 ? (
                      <EmptyState title="No matching lines" />
                    ) : (
                      filteredTranscript.map((t, i) => (
                        <div key={i} className="flex gap-3">
                          <span className="text-xs text-gray-400 w-16 shrink-0 pt-0.5">{t.time}</span>
                          <div>
                            <p className="text-sm font-semibold" style={{ color: t.color }}>
                              {t.speaker}
                            </p>
                            <p className="text-sm text-gray-700 mt-0.5">{t.text}</p>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                ) : (
                  <EmptyState
                    title="No transcript yet"
                    subtitle="Speaker-diarized transcript will appear here once the transcription pipeline processes this file."
                  />
                ))}

              {activeTab === 'Summary' &&
                (meeting.summary ? (
                  <div className="space-y-4">
                    <p className="text-sm text-gray-700 leading-relaxed">{meeting.summary}</p>
                    <div>
                      <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Key Decisions</h3>
                      <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                        {meeting.decisions.map((d, i) => (
                          <li key={i}>{d}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <EmptyState
                    title="No summary yet"
                    subtitle="An AI-generated executive summary, key decisions, and next steps will appear here after processing."
                  />
                ))}

              {activeTab === 'Insights' &&
                (meeting.keywords && meeting.keywords.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {meeting.keywords.map((k, i) => (
                      <span
                        key={i}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gray-50 border border-gray-100 text-xs font-medium text-gray-600"
                      >
                        {k.word}
                        <span className="text-gray-400">{k.count}</span>
                      </span>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No insights yet"
                    subtitle="Keyword extraction and topic insights will appear here after processing."
                  />
                ))}
            </div>
          </Card>

          <Card>
            <h2 className="font-semibold text-gray-900 dark:text-white mb-4">Speakers</h2>
            {meeting.speakerStats && meeting.speakerStats.length > 0 ? (
              <div className="space-y-4">
                {meeting.speakerStats.map((s, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-semibold text-gray-800 truncate">{s.name}</span>
                        <span className="text-gray-400 text-xs">{s.time}</span>
                      </div>
                      <div className="h-1.5 bg-gray-100 rounded-full mt-1.5 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${s.pct}%`, backgroundColor: s.color }}
                        />
                      </div>
                    </div>
                    <span className="text-xs text-gray-400 w-9 text-right">{s.pct}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState
                title="No speaker data yet"
                subtitle="Speaker diarization and talk-time breakdown will appear here after processing."
              />
            )}
          </Card>
        </div>

        {/* Right sidebar */}
        <div className="space-y-5">
          <Card className="bg-gradient-to-br from-primary-50/60 to-white">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={16} className="text-primary-600" />
              <h2 className="font-semibold text-gray-900 dark:text-white">AI Summary</h2>
            </div>
            {meeting.summary ? (
              <p className="text-sm text-gray-600 leading-relaxed">{meeting.summary}</p>
            ) : (
              <p className="text-sm text-gray-400">Not generated yet.</p>
            )}
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-4">
              <CheckCircle2 size={16} className="text-green-500" />
              <h2 className="font-semibold text-gray-900 dark:text-white">Action Items</h2>
            </div>
            {meeting.actionItems && meeting.actionItems.length > 0 ? (
              <div className="space-y-3">
                {meeting.actionItems.map((a, i) => (
                  <div key={i} className="flex items-start gap-3 pl-3 border-l-4 rounded" style={{ borderColor: a.color }}>
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-gray-800">{a.title}</p>
                      <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
                        <span>{a.assignee}</span>
                        <span className="flex items-center gap-1">
                          <Calendar size={11} /> {a.due}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-400">None yet.</p>
            )}
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-4">
              <Lightbulb size={16} className="text-amber-500" />
              <h2 className="font-semibold text-gray-900 dark:text-white">Decisions</h2>
            </div>
            {meeting.decisions && meeting.decisions.length > 0 ? (
              <ul className="list-disc list-inside text-sm text-gray-600 space-y-2">
                {meeting.decisions.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-400">None recorded yet.</p>
            )}
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-4">
              <Info size={16} className="text-primary-500" />
              <h2 className="font-semibold text-gray-900 dark:text-white">File Details</h2>
            </div>
            <div className="space-y-3 text-sm">
              <DetailRow label="Meeting ID" value={meeting.id} />
              <DetailRow label="File Name" value={meeting.fileName} />
              <DetailRow label="File Size" value={meeting.fileSizeLabel} />
              <DetailRow label="Uploaded" value={`${meeting.date} · ${meeting.time}`} />
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function MetaChip({ icon: Icon, children }) {
  return (
    <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-200 text-xs font-medium text-gray-600">
      <Icon size={13} />
      {children}
    </span>
  )
}

function DetailRow({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-gray-400 shrink-0">{label}</span>
      <span className="font-semibold text-gray-800 truncate">{value}</span>
    </div>
  )
}
