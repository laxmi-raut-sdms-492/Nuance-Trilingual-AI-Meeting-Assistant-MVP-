import { useEffect, useRef, useState, useMemo } from 'react'
import Icon from './Icon.jsx'
import { meetingsApi } from '../../services/api.js'

/**
 * Ported from the player in the design export (meeting_details_completed).
 *
 * The design shows a waveform. Decoding the real waveform would mean pulling
 * the whole file down and running it through an AudioContext — expensive for a
 * 300 MB recording and pointless before playback starts. Instead the bar
 * heights are derived deterministically from the meeting id, so a given
 * meeting always renders the same shape, and bars left of the playhead are
 * filled while the rest stay grey. The bar heights are decoration; the
 * playhead position, times and seeking are all real.
 *
 * Clicking a bar seeks, which the export's static markup only implies.
 */

const BAR_COUNT = 48

function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Deterministic pseudo-random heights from the id — stable across renders and
// reloads, so the waveform doesn't reshuffle on every state change.
function barHeights(seed = '') {
  let h = 0
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0
  return Array.from({ length: BAR_COUNT }, (_, i) => {
    h = (h * 1103515245 + 12345 + i) >>> 0
    return 20 + ((h >>> 8) % 71) // 20%..90%
  })
}

export default function AudioPlayer({ meetingId, fileName }) {
  const audioRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)
  const [failed, setFailed] = useState(false)

  const url = meetingId ? meetingsApi.audioUrl(meetingId) : null
  const heights = useMemo(() => barHeights(meetingId), [meetingId])

  useEffect(() => {
    setFailed(false)
    setPlaying(false)
    setCurrent(0)
    setDuration(0)
  }, [meetingId])

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) audio.pause()
    else audio.play().catch(() => setFailed(true))
  }

  const seekToFraction = (fraction) => {
    const audio = audioRef.current
    if (!audio || !duration) return
    const t = Math.min(Math.max(fraction, 0), 1) * duration
    audio.currentTime = t
    setCurrent(t)
  }

  if (!url || failed) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl bg-surface border border-border font-meta-data text-meta-data text-text-muted">
        <Icon name="audio_file" size={18} /> No audio available for this meeting.
      </div>
    )
  }

  const progress = duration ? current / duration : 0
  const playedBars = Math.round(progress * BAR_COUNT)

  return (
    <div
      className={`bg-surface border border-border rounded-xl p-4 flex items-center gap-4 md:gap-6 ${
        playing ? 'playing' : ''
      }`}
    >
      <audio
        ref={audioRef}
        src={url}
        preload="metadata"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
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

      {/* Waveform doubles as the scrubber. */}
      <div
        role="slider"
        tabIndex={0}
        aria-label="Seek"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(current)}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          seekToFraction((e.clientX - rect.left) / rect.width)
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') seekToFraction(progress + 0.02)
          if (e.key === 'ArrowLeft') seekToFraction(progress - 0.02)
        }}
        className="flex-1 flex items-center gap-1 h-10 cursor-pointer min-w-0"
      >
        {heights.map((h, i) => (
          <div
            key={i}
            className={`wave-bar flex-1 min-w-[2px] rounded-full origin-center transition-colors ${
              i < playedBars ? 'bg-primary-container active' : 'bg-border'
            }`}
            style={{ height: `${h}%` }}
          />
        ))}
      </div>

      <div className="font-meta-data text-meta-data text-text-muted whitespace-nowrap hidden sm:block">
        {formatTime(current)} / {formatTime(duration)}
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
  )
}
