import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import Icon from './Icon.jsx'
import { meetingsApi } from '../../services/api.js'
import { describeError } from '../../services/api.js'

/**
 * Ported from the player in the design export (meeting_details_completed).
 *
 * Waveform bar heights are decorative (seeded from meeting id). Playhead,
 * times, seeking, and the live "who is speaking" label are real — driven by
 * currentTime against transcript segment ranges.
 */

const BAR_COUNT = 48
const ACCEPTED = ['.mp3', '.wav', '.mp4', '.m4a', '.webm', '.ogg', '.flac', '.aac']
const MAX_UPLOAD_MB = 300

function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function barHeights(seed = '') {
  let h = 0
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0
  return Array.from({ length: BAR_COUNT }, (_, i) => {
    h = (h * 1103515245 + 12345 + i) >>> 0
    return 20 + ((h >>> 8) % 71)
  })
}

function speakerAtTime(segments, timeSec) {
  if (!segments?.length) return null
  const t = timeSec
  for (const seg of segments) {
    const start = Number(seg.start_sec)
    const end = Number(seg.end_sec)
    if (!isFinite(start) || !isFinite(end)) continue
    if (t >= start && t < end) return seg
  }
  let best = null
  for (const seg of segments) {
    const start = Number(seg.start_sec)
    if (isFinite(start) && start <= t) best = seg
  }
  return best
}

function NoAudioPanel({ onImportAudio, disabled }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [importing, setImporting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  const pickFile = useCallback(
    async (file) => {
      if (!file || !onImportAudio || disabled) return
      const ext = '.' + file.name.split('.').pop().toLowerCase()
      if (!ACCEPTED.includes(ext)) {
        setError(`Unsupported file type. Accepted: ${ACCEPTED.join(', ')}`)
        return
      }
      if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        setError(`File is over the ${MAX_UPLOAD_MB} MB limit.`)
        return
      }
      setError(null)
      setImporting(true)
      setProgress(0)
      try {
        await onImportAudio(file, (event) => {
          if (event.total) setProgress(Math.round((event.loaded / event.total) * 100))
        })
      } catch (err) {
        setError(describeError(err))
        setImporting(false)
        setProgress(0)
      }
    },
    [onImportAudio, disabled]
  )

  return (
    <div
      className={`rounded-xl border-2 border-dashed p-6 flex flex-col gap-4 transition-colors ${
        dragOver
          ? 'border-primary bg-primary/5'
          : 'border-border bg-surface'
      } ${disabled ? 'opacity-60 pointer-events-none' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        pickFile(e.dataTransfer.files?.[0])
      }}
    >
      <div className="flex items-start gap-3">
        <Icon name="audio_file" size={22} className="text-text-muted shrink-0 mt-0.5" />
        <div className="min-w-0">
          <p className="font-sidebar-header text-sidebar-header text-text-primary">
            No audio for this meeting
          </p>
          <p className="font-meta-data text-meta-data text-text-muted mt-1">
            Import a recording to play it back and run transcription.
          </p>
        </div>
      </div>

      {importing ? (
        <div className="flex flex-col gap-2">
          <div className="flex justify-between font-meta-data text-meta-data text-text-muted">
            <span>Uploading audio…</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-container progress-bar-fill rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="bg-primary-container text-on-primary-container px-4 py-2 rounded-lg font-label-sm text-label-sm hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            <Icon name="upload_file" className="text-[18px]" />
            Import audio
          </button>
          <span className="font-meta-data text-meta-data text-text-faint self-center">
            MP3, WAV, M4A, MP4
          </span>
        </div>
      )}

      {error && (
        <p className="font-meta-data text-meta-data text-error flex items-center gap-1">
          <Icon name="error" className="text-[16px]" />
          {error}
        </p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        className="hidden"
        onChange={(e) => pickFile(e.target.files?.[0])}
      />
    </div>
  )
}

const SPEED_OPTIONS = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

export default function AudioPlayer({
  meetingId,
  fileName,
  transcript = [],
  onImportAudio,
  importDisabled = false,
  audioVersion = 0,
  durationSeconds = 0,
}) {
  const audioRef = useRef(null)
  const speedMenuRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)
  const [failed, setFailed] = useState(false)
  const [checking, setChecking] = useState(true)
  const [hasAudio, setHasAudio] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [speedMenuOpen, setSpeedMenuOpen] = useState(false)

  const url = meetingId ? meetingsApi.audioUrl(meetingId) : null
  const heights = useMemo(() => barHeights(meetingId), [meetingId])

  const segments = useMemo(() => {
    return (transcript || [])
      .filter((t) => t && (t.speaker || t.speaker_label))
      .map((t) => ({
        start_sec: t.start_sec,
        end_sec: t.end_sec,
        speaker: t.speaker || t.speaker_label || 'Speaker',
        color: t.color || '#3b82f6',
      }))
      .sort((a, b) => Number(a.start_sec) - Number(b.start_sec))
  }, [transcript])

  const effectiveDuration = useMemo(() => {
    if (duration && isFinite(duration) && duration > 0) return duration
    if (durationSeconds && isFinite(durationSeconds) && durationSeconds > 0) return durationSeconds
    if (audioRef.current && isFinite(audioRef.current.duration) && audioRef.current.duration > 0) {
      return audioRef.current.duration
    }
    if (segments?.length > 0) {
      const last = segments[segments.length - 1]
      if (last && isFinite(Number(last.end_sec)) && Number(last.end_sec) > 0) return Number(last.end_sec)
    }
    return 0
  }, [duration, durationSeconds, segments])

  const active = useMemo(() => speakerAtTime(segments, current), [segments, current])

  useEffect(() => {
    function handleClickOutside(event) {
      if (speedMenuRef.current && !speedMenuRef.current.contains(event.target)) {
        setSpeedMenuOpen(false)
      }
    }
    if (speedMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [speedMenuOpen])

  useEffect(() => {
    setFailed(false)
    setPlaying(false)
    setCurrent(0)
    setDuration(0)
    setChecking(true)
    setHasAudio(Boolean(url))

    if (!url) {
      setChecking(false)
      setHasAudio(false)
      return
    }

    let cancelled = false
    fetch(url, { method: 'HEAD' })
      .then((res) => {
        if (cancelled) return
        setHasAudio(res.ok)
        if (!res.ok) setFailed(true)
      })
      .catch(() => {
        if (!cancelled) {
          if (fileName) {
            setHasAudio(true)
            setFailed(false)
          } else {
            setHasAudio(false)
            setFailed(true)
          }
        }
      })
      .finally(() => {
        if (!cancelled) setChecking(false)
      })

    return () => {
      cancelled = true
    }
  }, [meetingId, url, fileName, audioVersion])

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackSpeed
    }
  }, [playbackSpeed])

  const handleSpeedChange = (speed) => {
    setPlaybackSpeed(speed)
    if (audioRef.current) {
      audioRef.current.playbackRate = speed
    }
    setSpeedMenuOpen(false)
  }

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) audio.pause()
    else {
      audio.playbackRate = playbackSpeed
      audio.play().catch(() => setFailed(true))
    }
  }

  const seekToFraction = (fraction) => {
    const audio = audioRef.current
    if (!audio || !effectiveDuration) return
    const t = Math.min(Math.max(fraction, 0), 1) * effectiveDuration
    audio.currentTime = t
    setCurrent(t)
  }

  if (checking) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl bg-surface border border-border font-meta-data text-meta-data text-text-muted">
        <Icon name="hourglass_empty" size={18} className="animate-pulse" />
        Checking for audio…
      </div>
    )
  }

  if (!url || failed || !hasAudio) {
    if (onImportAudio) {
      return (
        <NoAudioPanel onImportAudio={onImportAudio} disabled={importDisabled} />
      )
    }
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl bg-surface border border-border font-meta-data text-meta-data text-text-muted">
        <Icon name="audio_file" size={18} /> No audio available for this meeting.
      </div>
    )
  }

  const progress = effectiveDuration > 0 ? Math.min(Math.max(current / effectiveDuration, 0), 1) : 0
  const playedBars = Math.round(progress * BAR_COUNT)

  return (
    <div className="bg-surface border border-border rounded-xl p-4 flex flex-col gap-3">
      <div className={`flex items-center gap-4 md:gap-6 ${playing ? 'playing' : ''}`}>
        <audio
          key={`${meetingId}-${audioVersion}`}
          ref={audioRef}
          src={url}
          preload="metadata"
          onLoadedMetadata={(e) => {
            setDuration(e.currentTarget.duration || 0)
            e.currentTarget.playbackRate = playbackSpeed
          }}
          onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
          onPlay={(e) => {
            setPlaying(true)
            e.currentTarget.playbackRate = playbackSpeed
          }}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onError={() => setFailed(true)}
        />

        <button
          type="button"
          onClick={togglePlay}
          aria-label={playing ? 'Pause' : 'Play'}
          className="w-12 h-12 flex-shrink-0 rounded-full bg-primary-container text-on-primary-container flex items-center justify-center hover:scale-105 transition-transform shadow-lg shadow-primary-container/20"
        >
          <Icon name={playing ? 'pause' : 'play_arrow'} filled className="text-[28px]" />
        </button>

        <div
          className="relative flex-1 flex items-center gap-1 h-10 cursor-pointer min-w-0 select-none group"
        >
          <input
            type="range"
            min={0}
            max={effectiveDuration || 1}
            step={0.1}
            value={current}
            onChange={(e) => {
              const val = parseFloat(e.target.value)
              if (audioRef.current) {
                audioRef.current.currentTime = val
              }
              setCurrent(val)
            }}
            aria-label="Seek audio timeline"
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-30"
          />

          {heights.map((h, i) => (
            <div
              key={i}
              className={`wave-bar flex-1 min-w-[2px] rounded-full origin-center transition-colors ${
                i < playedBars ? 'bg-primary-container active' : 'bg-border'
              }`}
              style={{ height: `${h}%` }}
            />
          ))}

          {/* Sliding Orange Thumb Handle */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-primary-container border-2 border-surface shadow-md pointer-events-none transition-transform group-hover:scale-125 z-20"
            style={{ left: `calc(${progress * 100}% - 8px)` }}
          />
        </div>

        <div className="font-meta-data text-meta-data text-text-muted whitespace-nowrap hidden sm:block">
          {formatTime(current)} / {formatTime(effectiveDuration)}
        </div>

        <a
          href={url}
          download={fileName || 'recording'}
          title="Download audio"
          aria-label="Download audio"
          className="text-text-muted hover:text-primary shrink-0 transition-colors"
        >
          <Icon name="download" size={20} />
        </a>
      </div>

      <div className="flex items-center justify-between min-h-[1.25rem] px-1 gap-2">
        <div className="flex items-center gap-2 min-w-0 overflow-hidden">
          {active ? (
            <>
              <span
                className="w-2 h-2 rounded-full shrink-0 animate-pulse"
                style={{ backgroundColor: active.color }}
                aria-hidden
              />
              <span
                className="font-label-sm text-label-sm uppercase tracking-wide truncate"
                style={{ color: active.color }}
              >
                {active.speaker}
              </span>
              <span className="font-meta-data text-meta-data text-text-muted whitespace-nowrap">
                speaking at {formatTime(current)}
              </span>
            </>
          ) : (
            <span className="font-meta-data text-meta-data text-text-faint truncate">
              {segments.length
                ? 'Seek or play to see who is speaking'
                : 'No speaker timeline for this recording'}
            </span>
          )}
        </div>

        {/* Playback Speed Control */}
        <div className="relative shrink-0" ref={speedMenuRef}>
          <button
            type="button"
            onClick={() => setSpeedMenuOpen((prev) => !prev)}
            title="Adjust playback speed"
            aria-label="Adjust playback speed"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-container-low hover:bg-surface-container-high border border-border/80 text-text-secondary hover:text-text-primary transition-all text-xs font-semibold shadow-xs"
          >
            <Icon name="speed" size={16} className="text-primary" />
            <span>{playbackSpeed}x</span>
            <Icon name={speedMenuOpen ? 'expand_less' : 'expand_more'} size={14} />
          </button>

          {speedMenuOpen && (
            <div className="absolute right-0 bottom-full mb-1.5 z-40 bg-surface border border-border rounded-xl shadow-xl p-1.5 flex flex-col min-w-[110px] animate-in fade-in zoom-in-95 duration-100">
              <div className="px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-text-muted border-b border-border/40 mb-1">
                Speed
              </div>
              {SPEED_OPTIONS.map((spd) => (
                <button
                  key={spd}
                  type="button"
                  onClick={() => handleSpeedChange(spd)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs text-left flex items-center justify-between transition-colors ${
                    playbackSpeed === spd
                      ? 'bg-primary-container text-on-primary-container font-bold'
                      : 'text-text-primary hover:bg-surface-container-high'
                  }`}
                >
                  <span>{spd}x</span>
                  {playbackSpeed === spd && <Icon name="check" size={14} />}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
