import { useParams, useNavigate } from 'react-router-dom'
import { useState, useEffect, useCallback, useRef } from 'react'
import toast from 'react-hot-toast'
import Icon from '../../components/common/Icon.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import Loader from '../../components/common/Loader.jsx'
import AudioPlayer from '../../components/common/AudioPlayer.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'
import { meetingsApi, speakersApi, describeError } from '../../services/api.js'

/**
 * Ported from the design export (meeting_details_completed),
 * meeting_details_processing/ and meeting_details_summary_tab/.
 *
 * This is the screen the product is really about: a trilingual, speaker-
 * diarized transcript that fills in live while the backend works.
 */

const TABS = ['Transcript', 'Summary', 'Insights']
const POLL_INTERVAL_MS = 3000

const STATUS_PILL = {
  Completed: { cls: 'bg-success/10 border-success/30 text-success', icon: 'check_circle' },
  Processing: { cls: 'bg-processing/10 border-processing/30 text-processing', icon: 'sync' },
  Failed: { cls: 'bg-error/10 border-error/30 text-error', icon: 'error' },
}

/** Devanagari needs its own face and looser leading than Latin at 16px. */
const isDevanagari = (code) => code === 'hi' || code === 'mr'

function isGenericSpeaker(name) {
  return /^speaker[_\s]?\d+$/i.test(String(name || '').trim())
}

/** True when a greeting names someone else but a later same-speaker line may be that person. */
function needsCollapsedGreetingRepair(transcript) {
  if (!transcript?.length) return false
  const greet =
    /(?:good\s+(?:morning|afternoon|evening)|hello|hi|namaste)\s+([A-Za-z][A-Za-z'-]+)/i
  const sorted = [...transcript].sort(
    (a, b) => (a.start_sec || 0) - (b.start_sec || 0),
  )
  const addressees = new Set()
  for (let i = 0; i < sorted.length; i++) {
    const match = String(sorted[i].text || '').match(greet)
    if (!match) continue
    const addressee = match[1]
    addressees.add(addressee.toLowerCase())
    const cur = sorted[i]
    for (let j = i + 1; j < sorted.length; j++) {
      const next = sorted[j]
      const gap = (next.start_sec || 0) - (cur.end_sec || cur.start_sec || 0)
      if (gap > 20) break
      const sameLabel =
        (next.speaker_label || next.speaker) === (cur.speaker_label || cur.speaker)
      const sameName =
        String(next.speaker || '').toLowerCase() === String(cur.speaker || '').toLowerCase()
      if (
        (sameLabel || sameName) &&
        next.speaker &&
        String(next.speaker).toLowerCase() !== addressee.toLowerCase()
      ) {
        return true
      }
    }
  }
  // Also repair when greetings mention people who never appear as speakers.
  if (addressees.size === 0) return false
  const speakerNames = new Set(
    sorted.map((t) => String(t.speaker || '').toLowerCase()).filter(Boolean),
  )
  for (const name of addressees) {
    if (!speakerNames.has(name)) return true
  }
  return false
}

function MetaChip({ icon, children }) {
  return (
    <div className="flex items-center gap-1.5 bg-surface border border-border px-2.5 py-1 rounded-md">
      <Icon name={icon} className="text-[16px]" />
      {children}
    </div>
  )
}

const SPEAKER_PALETTE = [
  '#3b82f6', // Blue
  '#10b981', // Green
  '#ef4444', // Red
  '#f59e0b', // Amber
  '#8b5cf6', // Purple
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#14b8a6', // Teal
  '#f97316', // Orange
  '#6366f1', // Indigo
]

function getSpeakerColor(name, speakerStats = []) {
  if (!name) return SPEAKER_PALETTE[0]
  const cleanName = String(name).trim()

  const found = speakerStats?.find(
    (s) => s.name?.toLowerCase() === cleanName.toLowerCase()
  )
  if (found && found.color) return found.color

  if (/^speaker[_\s]?\d+$/i.test(cleanName)) {
    const num = parseInt(cleanName.replace(/\D/g, ''), 10)
    if (!isNaN(num)) return SPEAKER_PALETTE[num % SPEAKER_PALETTE.length]
  }

  let hash = 0
  for (let i = 0; i < cleanName.length; i++) {
    hash = cleanName.charCodeAt(i) + ((hash << 5) - hash)
  }
  return SPEAKER_PALETTE[Math.abs(hash) % SPEAKER_PALETTE.length]
}

function resolveSpeakerDisplayName(rawName, speakerStats = []) {
  if (!rawName) return 'Speaker_00'
  const clean = String(rawName).trim()

  const statMatch = speakerStats?.find(
    (s) => (s.speaker_label && s.speaker_label.toLowerCase() === clean.toLowerCase()) ||
           (s.name && s.name.toLowerCase() === clean.toLowerCase())
  )
  if (statMatch && statMatch.name) return statMatch.name

  return clean
}

function deriveAttributedSpans(t, speakerStats) {
  if (t.attributed_spans && Array.isArray(t.attributed_spans) && t.attributed_spans.length > 0) {
    return t.attributed_spans.map((s) => {
      const resolved = resolveSpeakerDisplayName(s.speaker, speakerStats)
      return {
        speaker: resolved,
        text: s.text,
        color: getSpeakerColor(resolved, speakerStats),
      }
    })
  }

  if (!t.is_overlap) return null

  const uniqueSpks = getUniqueSpeakers(t.speaker, t.candidate_speakers, speakerStats)
  if (uniqueSpks.length <= 1) return null

  const fullText = (t.cleaned_text || t.text || '').trim()
  if (!fullText) return null

  const clauses = fullText.split(/(?<=[.!?])\s+|(?<=[,;])\s+/).filter(Boolean)
  const targetClauses = clauses.length > 0 ? clauses : [fullText]

  return targetClauses.map((clause, i) => {
    const rawSpk = uniqueSpks[i % uniqueSpks.length]
    const resolved = resolveSpeakerDisplayName(rawSpk, speakerStats)
    return {
      speaker: resolved,
      text: clause,
      color: getSpeakerColor(resolved, speakerStats),
    }
  })
}

function getUniqueSpeakers(speakerStr, candidateSpeakers, speakerStats = []) {
  let rawList = []
  if (Array.isArray(candidateSpeakers) && candidateSpeakers.length > 0) {
    rawList = candidateSpeakers
  } else if (speakerStr) {
    rawList = String(speakerStr).split(/\s*(?:\+|\&)\s*/).filter(Boolean)
  }

  const unique = []
  const seen = new Set()
  for (const item of rawList) {
    const resolved = resolveSpeakerDisplayName(item, speakerStats)
    const lower = resolved.toLowerCase()
    if (resolved && !seen.has(lower)) {
      seen.add(lower)
      unique.push(resolved)
    }
  }
  return unique.length > 0 ? unique : [resolveSpeakerDisplayName(speakerStr, speakerStats)]
}

function MultiSpeakerLabel({ meetingId, speakerStr, candidateSpeakers, fallbackColor, onRenamed, enrolledSpeakers, speakerStats }) {
  const parts = getUniqueSpeakers(speakerStr, candidateSpeakers)
  if (parts.length > 1) {
    return (
      <span className="inline-flex items-center gap-1 flex-wrap font-label-sm text-label-sm uppercase">
        {parts.map((p, idx) => {
          const spkColor = getSpeakerColor(p, speakerStats)
          return (
            <span key={idx} className="inline-flex items-center gap-1">
              {idx > 0 && <span className="text-text-muted text-[11px] font-bold normal-case">+</span>}
              <SpeakerLabel
                meetingId={meetingId}
                speaker={p}
                speakerLabel={p}
                color={spkColor}
                onRenamed={onRenamed}
                enrolledSpeakers={enrolledSpeakers}
              />
            </span>
          )
        })}
      </span>
    )
  }

  const singleName = parts[0] || speakerStr
  const spkColor = getSpeakerColor(singleName, speakerStats) || fallbackColor

  return (
    <SpeakerLabel
      meetingId={meetingId}
      speaker={singleName}
      speakerLabel={singleName}
      color={spkColor}
      onRenamed={onRenamed}
      enrolledSpeakers={enrolledSpeakers}
    />
  )
}

/**
 * Click a diarized label (e.g. "SPEAKER_00") to set a permanent name.
 * Saving always enrolls the voice so future meetings auto-label them.
 */
function SpeakerLabel({ meetingId, speaker, speakerLabel, color, onRenamed, enrolledSpeakers = [] }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(speaker)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setValue(speaker)
  }, [speaker])

  const cancel = () => {
    setEditing(false)
    setValue(speaker)
    setError('')
  }

  const commit = async (nameOverride, { permanent = true } = {}) => {
    const trimmed = (nameOverride ?? value).trim()
    if (!trimmed || trimmed === speaker) {
      cancel()
      return
    }
    setSaving(true)
    setError('')
    try {
      const key = speaker
      await meetingsApi.renameSpeaker(meetingId, key, trimmed, { remember: permanent })
      if (permanent) toast.success(`${trimmed} saved for all future meetings`)
      setEditing(false)
      onRenamed?.()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSaving(false)
    }
  }

  if (editing) {
    const suggestions = enrolledSpeakers.filter(
      (n) => n && n.toLowerCase() !== String(speaker).toLowerCase()
    )
    return (
      <span className="inline-flex flex-col items-start gap-1.5 max-w-[16rem]">
        <span className="inline-flex items-center gap-1 flex-wrap">
          <input
            autoFocus
            value={value}
            disabled={saving}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commit()
              if (e.key === 'Escape') cancel()
            }}
            placeholder="e.g. Anushka"
            className="font-label-sm text-label-sm uppercase bg-transparent border-b border-primary outline-none w-28"
            style={{ color }}
          />
          <button
            type="button"
            disabled={saving}
            onClick={() => commit()}
            className="font-meta-data text-meta-data text-primary hover:underline normal-case shrink-0"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={cancel}
            className="font-meta-data text-meta-data text-text-muted hover:underline normal-case shrink-0"
          >
            Cancel
          </button>
        </span>
        {suggestions.length > 0 && (
          <span className="flex flex-wrap gap-1 normal-case">
            {suggestions.map((name) => (
              <button
                key={name}
                type="button"
                disabled={saving}
                onClick={() => commit(name, { permanent: true })}
                className="px-2 py-0.5 rounded-md border border-border bg-surface-container-low font-meta-data text-meta-data text-text-primary hover:border-primary hover:text-primary transition-colors"
              >
                {name}
              </button>
            ))}
          </span>
        )}
        <span className="font-meta-data text-meta-data text-text-muted leading-snug normal-case">
          This name is saved permanently and used whenever this voice speaks again.
        </span>
        {error && <span className="text-[10px] text-error normal-case">{error}</span>}
      </span>
    )
  }

  return (
    <button
      type="button"
      title="Click to set a permanent speaker name"
      onClick={() => setEditing(true)}
      className="font-label-sm text-label-sm uppercase hover:underline decoration-dotted underline-offset-2"
      style={{ color }}
    >
      {speaker}
    </button>
  )
}

function SidePanel({ icon, iconClass = 'text-primary', title, action, children }) {
  return (
    <div className="bg-surface border border-border rounded-xl p-5 relative overflow-hidden">
      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-2xl -mr-10 -mt-10 pointer-events-none" />
      <div className="flex items-center justify-between gap-2 mb-4 relative z-10">
        <div className="flex items-center gap-2">
          <Icon name={icon} className={iconClass} />
          <h3 className="font-sidebar-header text-sidebar-header text-text-primary">{title}</h3>
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="relative z-10">{children}</div>
    </div>
  )
}

/**
 * Shown when the summarization stage produced nothing for this meeting.
 *
 * An empty panel is a real result, not a missing feature: the backend drops
 * every decision and action item whose quote it cannot find in the transcript,
 * so "none" means none survived verification. Saying that is the honest design —
 * filling the panel with plausible content is not.
 */
function NotGenerated({ what }) {
  return (
    <p className="font-meta-data text-meta-data text-text-muted leading-relaxed">
      {what} found nothing traceable.
    </p>
  )
}

/**
 * Provenance line under a generated summary.
 *
 * Decisions and action items are citation-checked against the transcript;
 * summary prose cannot be, because a paraphrase has no line to match. So the
 * engine that wrote it is named instead of leaving generated text looking
 * extracted. `extractive` is the fallback that only ever copies real lines.
 */

function SummaryProvenance({ engine }) {
  if (!engine) return null

  return (
    <p className="font-meta-data text-meta-data text-text-faint mt-3">
      {engine === 'extractive'
        ? 'Extracted verbatim from the transcript.'
        : 'AI-generated summary based on the meeting transcript.'}
    </p>
  )
}

/**
 * Normalises a decision entry to `{ text, quote, sourceTime }`.
 *
 * Decisions used to be plain strings; the backend now attaches the verified
 * transcript line each one came from. This keeps every render site working
 * whichever shape a given meeting's data happens to be in (old rows summarized
 * before evidence was tracked still come back as strings).
 */
function normalizeDecision(d) {
  return typeof d === 'string' ? { text: d, quote: null, sourceTime: null } : d
}

/**
 * The transcript line an insight was verified against, shown collapsed by
 * default in tight spaces (the side panel) and expanded where there is room
 * to make the point (the Insights tab) — see `defaultOpen`.
 *
 * This is the piece that makes a decision or action item traceable rather
 * than a bare claim: every one on screen survived a citation check against
 * this exact line (see models/summarizer.py:verify_quote), and the point of
 * showing it is to let the reader check that themselves instead of trusting
 * the extraction.
 */
function Evidence({ quote, time, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  if (!quote) return null
  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="font-meta-data text-meta-data text-text-faint hover:text-primary transition-colors flex items-center gap-1"
      >
        <Icon name={open ? 'expand_less' : 'format_quote'} className="text-[14px]" />
        {open ? 'Hide source line' : 'Show source line'}
      </button>
      {open && (
        <blockquote className="mt-1.5 pl-3 border-l-2 border-border font-meta-data text-meta-data text-text-muted italic leading-relaxed">
          {time && <span className="text-text-faint not-italic mr-1.5">[{time}]</span>}
          &ldquo;{quote}&rdquo;
        </blockquote>
      )}
    </div>
  )
}

/** Section header for the Insights tab — same visual language as SidePanel's
 * header, without the decorative background (this sits in the main column,
 * which already has its own card chrome). */
function InsightSection({ icon, iconClass = 'text-primary', title, children }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <Icon name={icon} className={iconClass} />
        <h3 className="font-sidebar-header text-sidebar-header text-text-primary">{title}</h3>
      </div>
      {children}
    </div>
  )
}

function ProcessingSkeleton() {
  return (
    <div className="flex flex-col gap-8" aria-hidden="true">
      {[80, 60, 90].map((w, i) => (
        <div key={i} className="flex gap-4">
          <div className="w-12 h-4 rounded skeleton-shimmer flex-shrink-0" />
          <div className="flex-1 flex flex-col gap-2">
            <div className="h-3 w-24 rounded skeleton-shimmer" />
            <div className="h-4 rounded skeleton-shimmer" style={{ width: `${w}%` }} />
            <div className="h-4 rounded skeleton-shimmer" style={{ width: `${w - 25}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function MeetingDetails() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { fetchMeeting } = useMeetings()

  // The list endpoint omits transcript bodies, so the full record is fetched
  // here rather than read out of the shared meetings array.
  const [meeting, setMeeting] = useState(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [activeTab, setActiveTab] = useState('Transcript')
  const [search, setSearch] = useState('')
  const [enrolledSpeakers, setEnrolledSpeakers] = useState([])
  const [audioVersion, setAudioVersion] = useState(0)
  const [showRawAsr, setShowRawAsr] = useState(false)
  const [rebuildingSummary, setRebuildingSummary] = useState(false)
  const autoLabeledRef = useRef(false)

  const load = useCallback(async () => {
    try {
      setMeeting(await fetchMeeting(id))
      setNotFound(false)
    } catch (err) {
      if (err?.response?.status === 404) {
        setMeeting(null)
        setNotFound(true)
      }
    } finally {
      setLoading(false)
    }
  }, [id, fetchMeeting])

  const loadEnrolled = useCallback(async () => {
    try {
      const { data } = await speakersApi.list()
      setEnrolledSpeakers(data.speakers || [])
    } catch {
      setEnrolledSpeakers([])
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    setMeeting(null)
    setNotFound(false)
    autoLabeledRef.current = false
    load()
    loadEnrolled()
  }, [load, loadEnrolled])

  // Keep refetching while the backend is still transcribing this meeting, so
  // the transcript fills in without the user reloading. This live-filling
  // transcript is the product demo — don't remove the polling.
  useEffect(() => {
    if (notFound || meeting?.status !== 'Processing') return
    const timer = setInterval(load, POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [notFound, meeting?.status, load])

  const handleImportAudio = useCallback(
    async (file, onUploadProgress) => {
      const { data } = await meetingsApi.uploadAudio(id, file, onUploadProgress)
      setMeeting(data)
      setAudioVersion((v) => v + 1)
      toast.success('Audio imported. Transcription is running.')
      return data
    },
    [id]
  )

  // Quietly auto-label Speaker_XX and repair collapsed greetings
  // (e.g. both lines named Vaishnavi when Lakshmi answered).
  useEffect(() => {
    if (!meeting || meeting.status !== 'Completed') return
    const hasGeneric =
      (meeting.speakerStats || []).some((s) => isGenericSpeaker(s.name)) ||
      (meeting.transcript || []).some((t) => isGenericSpeaker(t.speaker))
    const needsRepair =
      hasGeneric || needsCollapsedGreetingRepair(meeting.transcript)
    if (!needsRepair) {
      autoLabeledRef.current = false
      return
    }
    if (autoLabeledRef.current) return

    autoLabeledRef.current = true
    let cancelled = false
    ;(async () => {
      try {
        const { data } = await meetingsApi.identifySpeakers(id)
        if (cancelled) return
        if (data?.meeting && (data.applied || []).length > 0) {
          setMeeting(data.meeting)
          loadEnrolled()
          // Allow another pass if some Speaker_XX remain after partial apply.
          autoLabeledRef.current = false
        }
      } catch {
        // Stay silent — leave Speaker_XX if nothing could be resolved.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [meeting, id, loadEnrolled])

  if (loading) return <Loader label="Loading meeting..." />

  if (notFound || !meeting) {
    return (
      <EmptyState
        icon="search_off"
        title="Meeting not found"
        subtitle="It may have been deleted, or the link is incorrect."
        action={
          <button
            type="button"
            onClick={() => navigate('/meetings')}
            className="px-4 py-2 rounded-lg bg-cta hover:bg-primary-container text-on-cta font-label-sm text-label-sm transition-colors"
          >
            Back to meetings
          </button>
        }
      />
    )
  }

  const processing = meeting.status === 'Processing'
  const pill = STATUS_PILL[meeting.status] || STATUS_PILL.Completed
  const hasTranscript = meeting.transcript?.length > 0
  const filteredTranscript = hasTranscript
    ? meeting.transcript.filter((t) => {
        const hay = `${t.text || ''} ${t.raw_text || ''}`.toLowerCase()
        return hay.includes(search.toLowerCase())
      })
    : []

  return (
    <>
      {/* Header */}
      <div>
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-4">
          <div className="min-w-0">
            <div className="flex items-center gap-3 mb-2 flex-wrap">
              <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
                {meeting.title}
              </h2>
              <span
                className={`px-2 py-0.5 border font-label-sm text-label-sm rounded uppercase tracking-wider flex items-center gap-1 ${pill.cls}`}
              >
                <Icon
                  name={pill.icon}
                  className={`text-[14px] ${processing ? 'animate-spin' : ''}`}
                />
                {meeting.status}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2 mt-3 font-meta-data text-meta-data text-text-muted">
              <MetaChip icon="calendar_today">{meeting.date}</MetaChip>
              <MetaChip icon="schedule">{meeting.duration || meeting.time}</MetaChip>
              {(() => {
                const stats = meeting.speakerStats || []
                const totalSec = stats.reduce((n, s) => n + (s.seconds || 0), 0)
                const significant = stats.filter(
                  (s) =>
                    (s.seconds || 0) >= 3 ||
                    (totalSec > 0 && (s.seconds || 0) / totalSec >= 0.02),
                )
                const labelSet = new Set(
                  (meeting.transcript || [])
                    .map((t) => t.speaker_label || t.speaker)
                    .filter(Boolean),
                )
                const speakerCount =
                  significant.length ||
                  labelSet.size ||
                  meeting.participants ||
                  0
                return speakerCount > 0 ? (
                  <MetaChip icon="group">
                    {speakerCount} {speakerCount === 1 ? 'Speaker' : 'Speakers'}
                  </MetaChip>
                ) : null
              })()}
              {meeting.languages?.length > 0 && (
                <MetaChip icon="translate">
                  {meeting.languages.map((l) => l.code.toUpperCase()).join(', ')}
                </MetaChip>
              )}
              {meeting.processingMode && (
                <MetaChip icon={meeting.processingMode === 'cloud' ? 'cloud' : 'devices'}>
                  {meeting.processingMode === 'cloud'
                    ? `Cloud • ${
                        meeting.sttProvider
                          ? meeting.sttProvider.charAt(0).toUpperCase() + meeting.sttProvider.slice(1)
                          : 'Cloud'
                      }`
                    : 'Local Model'}
                </MetaChip>
              )}
              <MetaChip icon="audio_file">{meeting.fileSizeLabel}</MetaChip>
            </div>
          </div>

          <div className="flex gap-2 shrink-0">
            <button
              type="button"
              disabled={!hasTranscript}
              onClick={() => downloadTranscript(meeting)}
              className="bg-surface border border-border text-text-primary px-4 py-2 rounded-lg font-label-sm text-label-sm hover:bg-surface-raised transition-colors flex items-center gap-2 disabled:opacity-40 disabled:pointer-events-none"
            >
              <Icon name="download" className="text-[18px]" />
              Export
            </button>
          </div>
        </div>

        {meeting.agenda && (
          <p className="font-transcript-body text-transcript-body text-text-muted max-w-3xl">
            <span className="text-text-primary">Agenda:</span> {meeting.agenda}
          </p>
        )}
      </div>

      <AudioPlayer
        meetingId={meeting.id}
        fileName={meeting.fileName}
        transcript={meeting.transcript || []}
        onImportAudio={handleImportAudio}
        importDisabled={processing}
        audioVersion={audioVersion}
      />

      {processing && (
        <div className="bg-surface border border-processing/30 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-2">
            <Icon name="sync" className="text-processing animate-spin text-[20px]" />
            <p className="font-sidebar-header text-sidebar-header text-text-primary">Transcribing…</p>
          </div>
          <p className="font-meta-data text-meta-data text-text-muted">
            Detecting speech, separating speakers, and transcribing each segment in English, Hindi,
            or Marathi. This page updates itself as it goes.
            {hasTranscript && (
              <>
                {' '}
                <span className="text-processing">
                  {meeting.transcript.length} line
                  {meeting.transcript.length === 1 ? '' : 's'} so far.
                </span>
              </>
            )}
          </p>
          <div className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden mt-4">
            <div
              className="h-full bg-processing progress-bar-fill rounded-full pulse-amber"
              style={{ width: `${meeting.progress || 0}%` }}
            />
          </div>
          <p className="font-meta-data text-meta-data text-processing mt-1.5">
            {meeting.progress || 0}%
          </p>
        </div>
      )}

      {meeting.audioQualityWarning && (
        <div className="bg-processing/10 border border-processing/30 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-1">
            <Icon name="warning" className="text-processing" />
            <p className="font-sidebar-header text-sidebar-header text-text-primary">
              Audio quality notice
            </p>
          </div>
          <p className="font-meta-data text-meta-data text-text-muted">
            {meeting.audioQualityWarning}
          </p>
        </div>
      )}

      {meeting.status === 'Failed' && (
        <div className="bg-error/10 border border-error/20 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-1">
            <Icon name="error" className="text-error" />
            <p className="font-sidebar-header text-sidebar-header text-text-primary">
              Transcription failed
            </p>
          </div>
          <p className="font-meta-data text-meta-data text-text-muted">
            {meeting.error || 'The backend could not process this recording.'}
          </p>
        </div>
      )}

      {/* 2:1 grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main column */}
        <div className="lg:col-span-8 flex flex-col">
          <div className="flex items-center gap-6 border-b border-border mb-6 px-2 overflow-x-auto hide-scrollbar">
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setActiveTab(t)}
                className={`pb-3 border-b-2 font-label-sm text-label-sm uppercase tracking-widest transition-colors whitespace-nowrap ${
                  activeTab === t
                    ? 'border-primary-container text-primary'
                    : 'border-transparent text-text-muted hover:text-text-primary'
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          <div className="bg-surface border border-border rounded-xl p-6 flex flex-col gap-6">
            {activeTab === 'Transcript' && (
              <>
                {hasTranscript && (
                  <div className="flex flex-wrap items-center gap-4">
                    <div className="relative max-w-xs flex-1 min-w-[200px]">
                      <Icon
                        name="search"
                        size={16}
                        className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
                      />
                      <input
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search transcript..."
                        className="input-base w-full pl-9 pr-3 py-2 rounded-lg border font-meta-data text-meta-data placeholder:text-text-faint"
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowRawAsr((v) => !v)}
                      className={`px-3 py-2 rounded-lg border font-meta-data text-meta-data transition-colors ${
                        showRawAsr
                          ? 'border-primary-container text-primary bg-primary/5'
                          : 'border-border text-text-muted hover:text-text-primary'
                      }`}
                    >
                      {showRawAsr ? 'Hide raw ASR' : 'Show raw ASR'}
                    </button>
                  </div>
                )}

                {!hasTranscript && processing && <ProcessingSkeleton />}

                {!hasTranscript && !processing && (
                  <EmptyState
                    icon="record_voice_over"
                    title="No transcript"
                    subtitle="No speech was detected in this recording, or processing produced no usable lines."
                  />
                )}

                {hasTranscript && filteredTranscript.length === 0 && (
                  <EmptyState
                    icon="search_off"
                    title="No matching lines"
                    subtitle={`Nothing in this transcript matches "${search}".`}
                  />
                )}

                {filteredTranscript.length > 0 && (
                  <div className="flex flex-col gap-8">
                    {filteredTranscript.map((t, i) => {
                      const devanagari = isDevanagari(t.language)
                      return (
                        <div key={i} className="flex gap-4 group">
                          <div className="w-12 text-right flex-shrink-0 pt-1">
                            <span className="font-meta-data text-meta-data text-text-faint group-hover:text-text-muted transition-colors">
                              {t.time}
                            </span>
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                              <span className="flex items-center gap-1 shrink-0">
                                {(() => {
                                  const parts = getUniqueSpeakers(t.speaker, t.candidate_speakers)
                                  return parts.map((p, pIdx) => (
                                    <span
                                      key={pIdx}
                                      className="w-2 h-2 rounded-full shrink-0"
                                      style={{ backgroundColor: getSpeakerColor(p, meeting?.speakerStats) }}
                                    />
                                  ))
                                })()}
                              </span>
                              <MultiSpeakerLabel
                                meetingId={id}
                                speakerStr={t.speaker}
                                candidateSpeakers={t.candidate_speakers}
                                fallbackColor={t.color}
                                onRenamed={load}
                                enrolledSpeakers={enrolledSpeakers}
                                speakerStats={meeting?.speakerStats}
                              />
                              {/* Language is per line, not per meeting — a
                                  trilingual meeting switches mid-conversation. */}
                              {t.language && (
                                <span
                                  title={
                                    t.language_fallback
                                      ? `Detection was weak (${t.language_detected} @ ${t.language_prob}); fell back to the meeting's dominant language`
                                      : `${t.language_name} · confidence ${t.language_prob}`
                                  }
                                  className={`px-1.5 py-0.5 border rounded text-[10px] uppercase font-bold tracking-wider ml-2 ${
                                    t.language_fallback
                                      ? 'border-processing/40 text-processing'
                                      : 'border-border text-text-muted'
                                  }`}
                                >
                                  {t.language.toUpperCase()}
                                  {t.language_fallback ? '?' : ''}
                                </span>
                              )}
                              {t.language_mix && t.language_mix.length > 1 && (
                                <span
                                  title={`Mixed: ${t.language_mix.join(', ')}`}
                                  className="px-1.5 py-0.5 border border-border rounded text-[10px] text-text-muted ml-1"
                                >
                                  MIX
                                </span>
                              )}
                              {/* Distinct from MIX. MIX means this turn merged
                                  segments in different languages, each of which
                                  was transcribed correctly. This means ONE
                                  segment appears to hold two languages with no
                                  pause between them — it was transcribed as a
                                  single language by a single engine, so part of
                                  the text below is probably wrong. */}
                              {t.language_mixed_suspected && (
                                <span
                                  title={
                                    'Two languages scored almost equally here ' +
                                    `(margin ${t.language_margin}). This line was still ` +
                                    'transcribed as one language, so part of it may be inaccurate.'
                                  }
                                  className="px-1.5 py-0.5 border border-processing/40 rounded text-[10px] uppercase font-bold tracking-wider text-processing ml-1"
                                >
                                  Mixed?
                                </span>
                              )}
                              {t.is_overlap && (
                                <span
                                  title={
                                    'Overlapping Speech: Multiple speakers detected simultaneously ' +
                                    `(${ (t.candidate_speakers || [t.speaker]).join(' + ') }). Note: Speech was flagged as overlapping, not cleanly separated into per-speaker transcript lines.`
                                  }
                                  className="px-1.5 py-0.5 border border-amber-500/40 bg-amber-500/10 rounded text-[10px] uppercase font-bold tracking-wider text-amber-500 ml-1 inline-flex items-center gap-1"
                                >
                                  <span className="material-symbols-outlined text-[12px]">groups</span>
                                  Overlapping Speech
                                </span>
                              )}
                            </div>
                            {(() => {
                              const spans = deriveAttributedSpans(t, meeting?.speakerStats)
                              if (spans && spans.length > 0) {
                                return (
                                  <div className="flex flex-col gap-2 mt-2 border-l-2 border-amber-500/40 pl-3 py-1.5 bg-amber-500/5 rounded-r">
                                    {spans.map((span, sIdx) => {
                                      const spkColor = span.color || getSpeakerColor(span.speaker, meeting?.speakerStats)
                                      return (
                                        <div key={sIdx} className="flex items-start gap-2 text-sm">
                                          <span
                                            className="w-2.5 h-2.5 rounded-full shrink-0 mt-1.5"
                                            style={{ backgroundColor: spkColor }}
                                          />
                                          <span
                                            className="font-semibold text-xs shrink-0 mt-0.5"
                                            style={{ color: spkColor }}
                                          >
                                            {span.speaker}:
                                          </span>
                                          <span
                                            className={`px-2.5 py-1 rounded-md text-text-primary ${devanagari ? 'font-transcript-body-hi' : 'font-transcript-body'}`}
                                            style={{
                                              backgroundColor: `${spkColor}1F`,
                                              borderLeft: `3px solid ${spkColor}`
                                            }}
                                          >
                                            {span.text}
                                          </span>
                                        </div>
                                      )
                                    })}
                                  </div>
                                )
                              }
                              return (
                                <p
                                  lang={t.language}
                                  className={`text-text-primary ${
                                    devanagari
                                      ? 'font-transcript-body-hi text-transcript-body-hi'
                                      : 'font-transcript-body text-transcript-body'
                                  }`}
                                >
                                  {showRawAsr && t.raw_text ? t.raw_text : (t.cleaned_text || t.text)}
                                </p>
                              )
                            })()}
                            {showRawAsr && t.raw_text && t.cleaned_text && t.raw_text !== t.cleaned_text && (
                              <p className="mt-2 font-meta-data text-meta-data text-text-faint border-l-2 border-border pl-3">
                                Cleaned: {t.cleaned_text}
                              </p>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </>
            )}

            {activeTab === 'Summary' &&
              (meeting.summary ? (
                <>
                  <p className="font-transcript-body text-transcript-body text-text-primary">
                    {meeting.summary}
                  </p>
                  <SummaryProvenance engine={meeting.summaryEngine} />
                </>
              ) : (
                <EmptyState
                  icon="auto_awesome"
                  title="No summary available"
                  subtitle="There is no executive summary available from this transcript."
                />
              ))}

            {activeTab === 'Insights' &&
              (() => {
                /* This tab now carries two independently produced things, and
                   "empty" has to mean neither arrived. The top block is the
                   LLM's read of the meeting — what needs attention, what is
                   unresolved, who committed to what. The bottom block is the
                   citation-checked extraction, where every line shows the
                   transcript line it came from. One can land without the
                   other, so either alone is reason to render the tab. */
                const hasInsights =
                  meeting.insights?.attentionNeeded?.length > 0 ||
                  meeting.insights?.pending?.length > 0 ||
                  meeting.insights?.commitments?.length > 0 ||
                  meeting.keywords?.length > 0

                if (!hasInsights) {
                  return (
                    <EmptyState
                      icon="lightbulb"
                      title="No insights generated"
                      subtitle={
                        processing
                          ? 'Decisions, action items and keywords are extracted once transcription finishes.'
                          : "The summarization pass found nothing here — either no local model was reachable, or nothing it proposed could be verified against the transcript. This panel stays empty rather than showing invented content."
                      }
                    />
                  )
                }

                return (
                  <div className="flex flex-col gap-8">
                    {/* Meetings summarized before the insights pass existed
                        have no `insights` object at all. Their three cards
                        would render as "none" boxes, which reads as a finding
                        about the meeting rather than about the data, so the
                        block is gated on the pass having run. */}
                    {meeting.insights && (
                      <div className="flex flex-col gap-6">
                    {/* 🚨 Attention Needed */}
                    {meeting.insights?.attentionNeeded?.length > 0 ? (
                      <div className="bg-surface border border-error/30 rounded-xl p-5 shadow-sm">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span className="text-lg">🚨</span>
                            <h3 className="font-sidebar-header text-sidebar-header text-text-primary">Attention Needed</h3>
                          </div>
                          <span className="text-xs font-meta-data px-2.5 py-1 rounded-full bg-error/10 text-error border border-error/20">
                            {meeting.insights.attentionNeeded.length} item{meeting.insights.attentionNeeded.length > 1 ? 's' : ''} need attention
                          </span>
                        </div>
                        <div className="flex flex-col gap-2.5">
                          {meeting.insights.attentionNeeded.map((item, idx) => (
                            <div key={idx} className="flex items-start gap-2.5 font-meta-data text-meta-data text-text-muted">
                              <span className="mt-0.5 text-sm">{item.severity === 'red' ? '🔴' : '🟡'}</span>
                              <span className="leading-relaxed">{item.text}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="bg-surface border border-success/30 rounded-xl p-4 flex items-center justify-between shadow-sm">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">🟢</span>
                          <h3 className="font-sidebar-header text-sidebar-header text-text-primary">Attention Needed</h3>
                        </div>
                        <span className="text-xs font-meta-data text-success font-medium">No critical issues or attention items detected</span>
                      </div>
                    )}

                    {/* ⏳ Pending / Unresolved */}
                    {meeting.insights?.pending?.length > 0 ? (
                      <div className="bg-surface border border-border rounded-xl p-5 shadow-sm">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-lg">⏳</span>
                          <h3 className="font-sidebar-header text-sidebar-header text-text-primary">Pending / Unresolved</h3>
                        </div>
                        <p className="text-xs font-meta-data text-text-faint mb-4">Track things discussed that require follow-up resolution.</p>
                        <div className="flex flex-col gap-4">
                          {meeting.insights.pending.map((p, idx) => (
                            <div key={idx} className="flex flex-col gap-1.5 p-3.5 rounded-lg bg-surface-raised border border-border/60">
                              <div className="flex items-center justify-between">
                                <span className="font-semibold text-text-primary text-sm">{typeof p === 'string' ? p.slice(0, 40) : p.topic}</span>
                                <span className="text-xs font-meta-data px-2 py-0.5 rounded bg-warning/10 text-warning border border-warning/20">
                                  {typeof p === 'string' ? 'Pending' : (p.status || 'Pending')}
                                </span>
                              </div>
                              <p className="font-meta-data text-meta-data text-text-muted leading-relaxed">
                                {typeof p === 'string' ? p : p.description}
                              </p>
                              {typeof p === 'object' && p.owner && (
                                <span className="text-xs text-text-faint mt-1">Owner: <strong className="text-text-muted">{p.owner}</strong></span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="bg-surface border border-border rounded-xl p-4 flex items-center justify-between shadow-sm">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">⏳</span>
                          <h3 className="font-sidebar-header text-sidebar-header text-text-primary">Pending / Unresolved</h3>
                        </div>
                        <span className="text-xs font-meta-data text-text-faint">No pending or unresolved items recorded</span>
                      </div>
                    )}

                    {/* 📋 Action Items & Commitments */}
                    {meeting.insights?.commitments?.length > 0 ? (
                      <div className="bg-surface border border-border rounded-xl p-5 shadow-sm">
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-lg">📋</span>
                          <h3 className="font-sidebar-header text-sidebar-header text-text-primary">Action Items & Commitments</h3>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs font-meta-data">
                            <thead>
                              <tr className="border-b border-border text-text-faint uppercase">
                                <th className="py-2.5 px-3">Owner</th>
                                <th className="py-2.5 px-3">Action Item</th>
                                <th className="py-2.5 px-3">Timing / Deadline</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-border/60 text-text-muted">
                              {meeting.insights.commitments.map((c, idx) => (
                                <tr key={idx} className="hover:bg-surface-raised/50">
                                  <td className="py-3 px-3 font-semibold text-text-primary whitespace-nowrap">{c.owner || 'Team'}</td>
                                  <td className="py-3 px-3 leading-relaxed">{c.action || c.text}</td>
                                  <td className="py-3 px-3 whitespace-nowrap">
                                    <span className={`px-2 py-0.5 rounded text-xs ${c.timing && c.timing !== 'No explicit deadline stated' ? 'bg-primary/10 text-primary border border-primary/20' : 'bg-surface-raised text-text-faint'}`}>
                                      {c.timing || c.timeframe || 'No explicit deadline stated'}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-surface border border-border rounded-xl p-4 flex items-center justify-between shadow-sm">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">📋</span>
                          <h3 className="font-sidebar-header text-sidebar-header text-text-primary">Action Items & Commitments</h3>
                        </div>
                        <span className="text-xs font-meta-data text-text-faint">None. Nobody was assigned a specific task, deadline, or responsibility.</span>
                      </div>
                    )}
                      </div>
                    )}

                    <InsightSection icon="key" iconClass="text-primary" title="Keywords">
                      {meeting.keywords?.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {meeting.keywords.map((k) => (
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
                        <NotGenerated what="Keyword extraction" />
                      )}
                    </InsightSection>
                  </div>
                )
              })()}
          </div>
        </div>

        {/* Side panel */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <SidePanel icon="auto_awesome" title="AI Summary">
            {meeting.summary ? (
              <>
                <p className="font-meta-data text-meta-data text-text-muted leading-relaxed">
                  {meeting.summary}
                </p>
                <SummaryProvenance engine={meeting.summaryEngine} />
              </>
            ) : (
              <p className="font-meta-data text-meta-data text-text-muted leading-relaxed">
                There is no executive summary available from this transcript.
              </p>
            )}
          </SidePanel>

          <SidePanel icon="group" title="Speakers">
            {(() => {
              const cleanStats = (() => {
                if (!meeting.speakerStats?.length) return []
                const totals = {}
                for (const s of meeting.speakerStats) {
                  const name = String(s.name || '').trim()
                  if (!name) continue
                  const secs = parseFloat(s.seconds || 0)
                  if (secs <= 0) continue

                  const parts = name.split(/\s*(?:\+|\&)\s*/).filter(Boolean)
                  const share = secs / Math.max(parts.length, 1)
                  for (const p of parts) {
                    totals[p] = (totals[p] || 0) + share
                  }
                }
                const grandTotal = Object.values(totals).reduce((a, b) => a + b, 0)
                if (grandTotal <= 0) return []

                return Object.entries(totals)
                  .map(([name, secs]) => ({
                    name,
                    seconds: Math.round(secs * 10) / 10,
                    pct: Math.round((secs / grandTotal) * 1000) / 10,
                    color: getSpeakerColor(name, meeting.speakerStats),
                  }))
                  .sort((a, b) => b.seconds - a.seconds)
              })()

              if (!cleanStats.length) {
                return processing ? (
                  <p className="font-meta-data text-meta-data text-text-muted">
                    Speaker breakdown appears as segments are processed.
                  </p>
                ) : (
                  <p className="font-meta-data text-meta-data text-text-muted">
                    No speakers were detected in this recording.
                  </p>
                )
              }

              return (
                <div className="flex flex-col gap-4">
                  {cleanStats.map((s) => (
                    <div key={s.name} className="flex items-center gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between font-meta-data text-meta-data">
                          <span className="text-text-primary truncate">{s.name}</span>
                          <span className="text-text-faint">{s.seconds}s</span>
                        </div>
                        <div className="h-1.5 bg-surface-container-high rounded-full mt-1.5 overflow-hidden">
                          <div
                            className="h-full rounded-full progress-bar-fill"
                            style={{ width: `${s.pct}%`, backgroundColor: s.color }}
                          />
                        </div>
                      </div>
                      <span className="font-meta-data text-meta-data text-text-faint w-9 text-right">
                        {s.pct}%
                      </span>
                    </div>
                  ))}
                </div>
              )
            })()}
          </SidePanel>

          <SidePanel icon="translate" title="Languages">
            {meeting.languages?.length > 0 ? (
              <div className="flex flex-col gap-3">
                {meeting.languages.map((l) => (
                  <div key={l.code}>
                    <div className="flex items-center justify-between font-meta-data text-meta-data">
                      <span className="text-text-primary">{l.name}</span>
                      <span className="text-text-faint">{l.pct}%</span>
                    </div>
                    <div className="h-1.5 bg-surface-container-high rounded-full mt-1.5 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary-container progress-bar-fill"
                        style={{ width: `${l.pct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="font-meta-data text-meta-data text-text-muted">
                Detected per segment once transcription runs.
              </p>
            )}
          </SidePanel>

          <SidePanel icon="task_alt" iconClass="text-success" title="Action Items">
            {meeting.actionItems?.length > 0 ? (
              <div className="flex flex-col gap-3">
                {meeting.actionItems.map((a, i) => (
                  <div
                    key={i}
                    className="pl-3 border-l-2"
                    style={{ borderColor: a.color || 'var(--color-border)' }}
                  >
                    <p className="text-text-primary">{a.title}</p>
                    {/* Assembled from what exists. An unverified assignee and an
                        unstated due date both come back null — rendering the
                        separator anyway left a bare "·" implying missing data. */}
                    {(a.assignee || a.due) && (
                      <p className="font-meta-data text-meta-data text-text-muted mt-1">
                        {[a.assignee, a.due].filter(Boolean).join(' · ')}
                      </p>
                    )}
                    <Evidence quote={a.quote} time={a.sourceTime} />
                  </div>
                ))}
              </div>
            ) : (
              <p className="font-meta-data text-meta-data text-text-muted leading-relaxed">
                None. Nobody was assigned a specific task, deadline, or responsibility.
              </p>
            )}
          </SidePanel>

          <SidePanel icon="gavel" iconClass="text-processing" title="Key Decisions">
            {meeting.decisions?.length > 0 ? (
              <ul className="list-disc list-inside font-meta-data text-meta-data text-text-muted flex flex-col gap-3">
                {meeting.decisions.map((d, i) => {
                  const decision = normalizeDecision(d)
                  return (
                    <li key={i}>
                      {decision.text}
                      <Evidence quote={decision.quote} time={decision.sourceTime} />
                    </li>
                  )
                })}
              </ul>
            ) : (
              <p className="font-meta-data text-meta-data text-text-muted leading-relaxed">
                None. No formal decisions were recorded in this conversation.
              </p>
            )}
          </SidePanel>

          <SidePanel icon="info" title="File Details">
            <div className="flex flex-col gap-3 font-meta-data text-meta-data">
              <DetailRow label="Meeting ID" value={meeting.id} />
              <DetailRow label="File Name" value={meeting.fileName} />
              <DetailRow label="File Size" value={meeting.fileSizeLabel} />
              <DetailRow label="Uploaded" value={`${meeting.date} · ${meeting.time}`} />
              {meeting.failedSegments > 0 && (
                <DetailRow
                  label="Failed segments"
                  value={String(meeting.failedSegments)}
                  tone="text-processing"
                />
              )}
            </div>
          </SidePanel>
        </div>
      </div>
    </>
  )
}

// Plain-text export of what the pipeline actually produced. Speaker, timestamp
// and detected language per line, because a Marathi line and an English line
// look identical in a bare transcript once the audio is gone.
function downloadTranscript(meeting) {
  const header = [
    meeting.title,
    `Uploaded: ${meeting.date} ${meeting.time}`,
    meeting.agenda ? `Agenda: ${meeting.agenda}` : null,
    meeting.languages?.length
      ? `Languages: ${meeting.languages.map((l) => `${l.name} ${l.pct}%`).join(', ')}`
      : null,
    '',
  ]
    .filter(Boolean)
    .join('\n')

  const body = meeting.transcript
    .map((t) => `[${t.time}] ${t.speaker} (${t.language_name || '—'}): ${t.text}`)
    .join('\n')

  const blob = new Blob([`${header}\n${body}\n`], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${meeting.title.replace(/[^\w\d]+/g, '-').toLowerCase()}-transcript.txt`
  link.click()
  URL.revokeObjectURL(url)
}

function DetailRow({ label, value, tone = 'text-text-primary' }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-text-faint shrink-0">{label}</span>
      <span className={`${tone} truncate`} title={value}>
        {value}
      </span>
    </div>
  )
}
