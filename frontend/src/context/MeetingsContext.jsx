import { createContext, useContext, useEffect, useState, useCallback, useMemo, useRef } from 'react'
import { meetingsApi, describeError } from '../services/api.js'

const MeetingsContext = createContext(null)

// While anything is still transcribing, re-fetch the list on this interval so
// status and progress move without the user reloading. Polling stops the
// moment nothing is in "Processing".
const POLL_INTERVAL_MS = 3000

// The backend stores one UTC timestamp per meeting and deliberately does no
// date formatting — the server's timezone is not the viewer's. These derived
// fields are what the list and detail screens render.
export function normalizeMeeting(meeting) {
  const uploadedAt = meeting.uploadedAtISO ? new Date(meeting.uploadedAtISO) : null
  return {
    ...meeting,
    date: uploadedAt
      ? uploadedAt.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
      : '—',
    time: uploadedAt
      ? uploadedAt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
      : '—'
  }
}

export function MeetingsProvider({ children }) {
  const [meetings, setMeetings] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Guards against a slow in-flight refresh overwriting newer state. Any
  // local mutation bumps this, so a poll that was already on the wire when
  // the user deleted or uploaded something is discarded instead of
  // resurrecting the old list.
  const requestSeq = useRef(0)
  const invalidateInFlight = useCallback(() => ++requestSeq.current, [])

  const refresh = useCallback(async ({ quiet = false } = {}) => {
    const seq = ++requestSeq.current
    if (!quiet) setLoading(true)
    try {
      const { data } = await meetingsApi.list()
      if (seq !== requestSeq.current) return
      setMeetings((data.meetings || []).map(normalizeMeeting))
      setError(null)
    } catch (err) {
      if (seq !== requestSeq.current) return
      setError(describeError(err))
    } finally {
      if (seq === requestSeq.current && !quiet) setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const hasProcessing = useMemo(
    () => meetings.some((m) => m.status === 'Processing'),
    [meetings]
  )

  useEffect(() => {
    if (!hasProcessing) return
    const id = setInterval(() => refresh({ quiet: true }), POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [hasProcessing, refresh])

  // Uploads the real file to the backend, which transcribes it in the
  // background. Resolves as soon as the record exists (status "Processing") —
  // the poll above then carries it to "Completed".
  const addMeeting = useCallback(async (file, details = {}, onUploadProgress) => {
    const formData = new FormData()
    formData.append('file', file, file.name)
    formData.append('title', details.title?.trim() || '')
    formData.append('agenda', details.agenda?.trim() || '')
    formData.append('stt_adapter', details.stt_adapter || 'local')

    // Backward compatible: any existing caller that doesn't pass
    // processingMode still sends 'local', which is also the backend's own
    // default for an omitted/blank value — see stt/resolver.py.
    const processingMode = details.processingMode === 'cloud' ? 'cloud' : 'local'
    formData.append('processing_mode', processingMode)
    if (processingMode === 'cloud') {
      formData.append('stt_provider', details.sttProvider || 'sarvam')
    }

    const { data } = await meetingsApi.create(formData, onUploadProgress)
    const record = normalizeMeeting(data)
    invalidateInFlight()
    setMeetings((prev) => [record, ...prev.filter((m) => m.id !== record.id)])
    return record
  }, [invalidateInFlight])

  const removeMeeting = useCallback(async (id) => {
    invalidateInFlight()
    setMeetings((prev) => prev.filter((m) => m.id !== id)) // optimistic
    try {
      await meetingsApi.remove(id)
    } catch (err) {
      // The delete didn't happen. Resync from the server rather than trying
      // to restore a snapshot that may already be stale.
      refresh({ quiet: true })
      throw new Error(describeError(err))
    }
  }, [invalidateInFlight, refresh])

  // List responses omit transcript bodies (see the backend's _summarize), so
  // the detail page fetches the full record separately.
  const fetchMeeting = useCallback(async (id) => {
    const { data } = await meetingsApi.getById(id)
    return normalizeMeeting(data)
  }, [])

  const getById = useCallback((id) => meetings.find((m) => m.id === id), [meetings])

  return (
    <MeetingsContext.Provider
      value={{ meetings, loading, error, refresh, addMeeting, removeMeeting, getById, fetchMeeting }}
    >
      {children}
    </MeetingsContext.Provider>
  )
}

export function useMeetings() {
  const ctx = useContext(MeetingsContext)
  if (!ctx) throw new Error('useMeetings must be used within MeetingsProvider')
  return ctx
}
