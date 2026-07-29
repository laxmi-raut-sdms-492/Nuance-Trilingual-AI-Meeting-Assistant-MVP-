import { useEffect, useRef, useState } from 'react'
import { Mic, Square, Pause, Play } from 'lucide-react'

function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Records audio from the mic and hands the finished Blob back to the parent
// as soon as "Stop & Save" is pressed - the parent then saves it as a
// meeting automatically, so nothing further needs to be clicked.
export default function AudioRecorder({ onRecordingComplete, disabled }) {
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

  const startTimer = () => {
    timerRef.current = setInterval(() => {
      secondsRef.current += 1
      setSeconds(secondsRef.current)
    }, 1000)
  }
  const stopTimer = () => clearInterval(timerRef.current)

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
      setError('Microphone access was denied or is unavailable. Please allow microphone permission and try again.')
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

  const stop = () => {
    setStatus('saving')
    stopTimer()
    mediaRecorderRef.current?.stop()
  }

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16">
      <div
        className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors ${
          status === 'recording' ? 'bg-red-50' : 'bg-primary-50'
        }`}
      >
        <div
          className={`w-14 h-14 rounded-full flex items-center justify-center ${
            status === 'recording' ? 'bg-red-500 animate-pulse' : 'bg-primary-600'
          }`}
        >
          <Mic size={22} className="text-white" />
        </div>
      </div>

      <p className="text-2xl font-bold text-gray-900 dark:text-white tabular-nums">{formatTime(seconds)}</p>
      <p className="text-xs text-gray-400 text-center">
        {status === 'idle' && 'Click start to begin recording this meeting'}
        {status === 'recording' && 'Recording in progress...'}
        {status === 'paused' && 'Recording paused'}
        {status === 'saving' && 'Saving recording...'}
      </p>

      {error && <p className="text-xs text-red-500 max-w-sm text-center">{error}</p>}

      <div className="flex items-center gap-3">
        {status === 'idle' && (
          <button
            onClick={start}
            disabled={disabled}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary-600 text-white text-sm font-semibold hover:bg-primary-700 disabled:opacity-50"
          >
            <Mic size={16} /> Start Recording
          </button>
        )}
        {status === 'recording' && (
          <>
            <button
              onClick={pause}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-gray-200 text-sm font-semibold text-gray-700 hover:bg-gray-50"
            >
              <Pause size={16} /> Pause
            </button>
            <button
              onClick={stop}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-red-500 text-white text-sm font-semibold hover:bg-red-600"
            >
              <Square size={14} /> Stop & Save
            </button>
          </>
        )}
        {status === 'paused' && (
          <>
            <button
              onClick={resume}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-gray-200 text-sm font-semibold text-gray-700 hover:bg-gray-50"
            >
              <Play size={16} /> Resume
            </button>
            <button
              onClick={stop}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-red-500 text-white text-sm font-semibold hover:bg-red-600"
            >
              <Square size={14} /> Stop & Save
            </button>
          </>
        )}
      </div>

      <p className="text-[11px] text-gray-300 max-w-xs text-center mt-1">
        The recording is saved automatically as soon as you stop — it will appear in All Meetings.
      </p>
    </div>
  )
}
