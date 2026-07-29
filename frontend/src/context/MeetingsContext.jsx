import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { deleteAudioBlob } from '../utils/audioStore.js'

const MeetingsContext = createContext(null)
const STORAGE_KEY = 'meetiq:meetings'

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveToStorage(meetings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(meetings))
  } catch {
    // storage full or unavailable - fail silently, state still works in-memory
  }
}

function formatBytes(bytes) {
  if (!bytes) return '0 MB'
  const mb = bytes / (1024 * 1024)
  if (mb > 1024) return `${(mb / 1024).toFixed(2)} GB`
  return `${mb.toFixed(2)} MB`
}

function niceTitleFromFileName(name) {
  const withoutExt = name.replace(/\.[^/.]+$/, '')
  return withoutExt
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase()) || 'Untitled Meeting'
}

export function MeetingsProvider({ children }) {
  const [meetings, setMeetings] = useState(() => loadFromStorage())

  useEffect(() => {
    saveToStorage(meetings)
  }, [meetings])

  // Adds a real uploaded file as a meeting record. No fake transcript/summary
  // is generated - those fields stay empty until the FSD backend (Replicate +
  // OpenRouter pipeline) is connected and returns real results.
  const addMeeting = useCallback((file, details = {}) => {
    const id = `MTG-${Date.now()}`
    const now = new Date()
    const record = {
      id,
      title: details.title?.trim() || niceTitleFromFileName(file.name),
      agenda: details.agenda?.trim() || null,
      fileName: file.name,
      fileType: file.type || 'unknown',
      fileSizeLabel: formatBytes(file.size),
      fileSizeBytes: file.size,
      date: now.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' }),
      time: now.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }),
      uploadedAtISO: now.toISOString(),
      status: 'Processing',
      duration: null,
      participants: null,
      language: null,
      organizer: null,
      location: null,
      summary: null,
      decisions: [],
      actionItems: [],
      speakerStats: [],
      keywords: [],
      transcript: []
    }
    setMeetings((prev) => [record, ...prev])
    return record
  }, [])

  const removeMeeting = useCallback((id) => {
    setMeetings((prev) => prev.filter((m) => m.id !== id))
    deleteAudioBlob(id)
  }, [])

  const getById = useCallback((id) => meetings.find((m) => m.id === id), [meetings])

  const markCompleted = useCallback((id, updates = {}) => {
    setMeetings((prev) =>
      prev.map((m) => (m.id === id ? { ...m, status: 'Completed', ...updates } : m))
    )
  }, [])

  return (
    <MeetingsContext.Provider
      value={{ meetings, addMeeting, removeMeeting, getById, markCompleted }}
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
