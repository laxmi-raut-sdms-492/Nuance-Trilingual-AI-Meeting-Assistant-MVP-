import { useEffect, useRef, useState } from 'react'
import Icon from './Icon.jsx'

function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

/**
 * Records from the mic and hands the finished Blob to the parent on stop.
 *
 * Restyled against the design export; the recording logic is unchanged
 * apart from `maxSeconds`, added so speaker enrollment can auto-stop at the
 * 6-second sample the design asks for instead of relying on the user.
 */
export default function AudioRecorder({
  onRecordingComplete,
  disabled,
  maxSeconds,
  hint = 'The recording is saved automatically as soon as you stop — it will appear in All Meetings.',
}) {
  const [status, setStatus] = useState('idle') // idle | recording | paused | saving
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState('')
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const secondsRef = useRef(0)

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const stopTimer = () => clearInterval(timerRef.current)

  const stop = () => {
    setStatus('saving')
    stopTimer()
    mediaRecorderRef.current?.stop()
  }

  const startTimer = () => {
    timerRef.current = setInterval(() => {
      secondsRef.current += 1
      setSeconds(secondsRef.current)
      // Auto-stop for fixed-length samples (enrollment).
      if (maxSeconds && secondsRef.current >= maxSeconds) stop()
    }, 1000)
  }

  const start = async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Recording is not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : ''
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        streamRef.current?.getTracks().forEach((t) => t.stop())
        setStatus('idle')
        stopTimer()
        onRecordingComplete(blob, secondsRef.current)
        secondsRef.current = 0
        setSeconds(0)
      }

      recorder.start()
      mediaRecorderRef.current = recorder
      secondsRef.current = 0
      setSeconds(0)
      setStatus('recording')
      startTimer()
    } catch {
      setError(
        'Microphone access was denied or is unavailable. Allow microphone permission and try again.'
      )
    }
  }

  const pause = () => {
    mediaRecorderRef.current?.pause()
    setStatus('paused')
    stopTimer()
  }

  const resume = () => {
    mediaRecorderRef.current?.resume()
    setStatus('recording')
    startTimer()
  }

  const recording = status === 'recording'
  const secondary =
    'flex items-center gap-2 px-4 py-2.5 rounded-lg border border-border text-text-primary font-label-sm text-label-sm hover:bg-surface-raised transition-colors'
  const stopBtn =
    'flex items-center gap-2 px-4 py-2.5 rounded-lg bg-error text-on-error font-label-sm text-label-sm hover:bg-error/90 transition-colors'

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10">
      <div
        className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors ${
          recording ? 'bg-error/10' : 'bg-surface-raised border border-border'
        }`}
      >
        <div
          className={`w-14 h-14 rounded-full flex items-center justify-center transition-colors ${
            recording ? 'bg-error animate-pulse' : 'bg-primary-container'
          }`}
        >
          <Icon
            name="mic"
            filled
            className={recording ? 'text-on-error' : 'text-on-primary-container'}
          />
        </div>
      </div>

      <p className="font-headline-lg text-headline-lg text-text-primary tabular-nums">
        {formatTime(seconds)}
        {maxSeconds ? (
          <span className="text-text-faint"> / {formatTime(maxSeconds)}</span>
        ) : null}
      </p>

      <p className="font-meta-data text-meta-data text-text-muted text-center">
        {status === 'idle' && 'Press start to begin recording'}
        {status === 'recording' && 'Recording in progress...'}
        {status === 'paused' && 'Recording paused'}
        {status === 'saving' && 'Saving recording...'}
      </p>

      {error && (
        <p className="font-meta-data text-meta-data text-error max-w-sm text-center">{error}</p>
      )}

      <div className="flex items-center gap-3">
        {status === 'idle' && (
          <button
            type="button"
            onClick={start}
            disabled={disabled}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-cta text-on-cta font-label-sm text-label-sm hover:bg-primary-container transition-colors disabled:opacity-40"
          >
            <Icon name="mic" size={18} /> Start Recording
          </button>
        )}
        {recording && (
          <>
            {!maxSeconds && (
              <button type="button" onClick={pause} className={secondary}>
                <Icon name="pause" size={18} /> Pause
              </button>
            )}
            <button type="button" onClick={stop} className={stopBtn}>
              <Icon name="stop" size={18} /> Stop &amp; Save
            </button>
          </>
        )}
        {status === 'paused' && (
          <>
            <button type="button" onClick={resume} className={secondary}>
              <Icon name="play_arrow" size={18} /> Resume
            </button>
            <button type="button" onClick={stop} className={stopBtn}>
              <Icon name="stop" size={18} /> Stop &amp; Save
            </button>
          </>
        )}
      </div>

      {hint && (
        <p className="font-meta-data text-meta-data text-text-faint max-w-xs text-center">{hint}</p>
      )}
    </div>
  )
}
