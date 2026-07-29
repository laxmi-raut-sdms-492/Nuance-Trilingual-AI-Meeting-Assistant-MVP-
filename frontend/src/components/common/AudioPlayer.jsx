import { useEffect, useRef, useState } from 'react'
import { Play, Pause, Download, FileAudio } from 'lucide-react'
import { getAudioBlob } from '../../utils/audioStore.js'

function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function AudioPlayer({ meetingId, fileName }) {
  const audioRef = useRef(null)
  const [url, setUrl] = useState(null)
  const [loading, setLoading] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    let objectUrl
    let cancelled = false
    setLoading(true)
    setUrl(null)

    getAudioBlob(meetingId).then((blob) => {
      if (cancelled) return
      if (blob) {
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
      }
      setLoading(false)
    })

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [meetingId])

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) audio.pause()
    else audio.play()
  }

  const handleSeek = (e) => {
    const audio = audioRef.current
    const value = Number(e.target.value)
    if (audio) audio.currentTime = value
    setCurrent(value)
  }

  if (loading) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl bg-gray-50 border border-gray-100 text-sm text-gray-400">
        <FileAudio size={16} className="animate-pulse" /> Loading audio...
      </div>
    )
  }

  if (!url) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-xl bg-gray-50 border border-gray-100 text-sm text-gray-400">
        <FileAudio size={16} /> No audio available for this meeting yet.
      </div>
    )
  }

  return (
    <div className="p-4 rounded-xl bg-gray-50 border border-gray-100">
      <audio
        ref={audioRef}
        src={url}
        preload="metadata"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="w-10 h-10 rounded-full bg-primary-600 text-white flex items-center justify-center shrink-0 hover:bg-primary-700 transition-colors"
        >
          {playing ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
        </button>
        <div className="flex-1 min-w-0">
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={current}
            onChange={handleSeek}
            className="w-full accent-primary-600"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>{formatTime(current)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
        <a
          href={url}
          download={fileName || 'recording'}
          className="text-gray-400 hover:text-primary-600 shrink-0"
          title="Download audio"
        >
          <Download size={16} />
        </a>
      </div>
    </div>
  )
}
