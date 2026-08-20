import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000
})

// Turns an axios failure into something worth showing a user. Without this
// every error surfaces as the string "Network Error", which doesn't tell
// anyone that the Python backend simply isn't running.
export function describeError(error) {
  if (error?.response?.data?.detail) return error.response.data.detail
  if (error?.response) return `Server returned ${error.response.status}.`
  if (error?.code === 'ECONNABORTED') return 'The request timed out.'
  return `Cannot reach the API at ${API_BASE_URL}. Is the backend running?`
}

export const meetingsApi = {
  list: () => api.get('/meetings'),

  getById: (id) => api.get(`/meetings/${id}`),

  // Uploads get no timeout: a 200 MB recording on a slow connection will
  // exceed any fixed limit, and aborting a nearly-finished upload is worse
  // than waiting. Progress is reported instead so the UI stays responsive.
  create: (formData, onUploadProgress) =>
    api.post('/meetings', formData, {
      timeout: 0,
      onUploadProgress
    }),

  remove: (id) => api.delete(`/meetings/${id}`),
  trash: () => api.get('/meetings/trash'),
  purgeAll: () => api.delete('/meetings/trash'),
  restore: (id) => api.post(`/meetings/${id}/restore`),
  purge: (id) => api.delete(`/meetings/${id}/purge`),

  search: (query) => api.get('/search', { params: { q: query } }),

  languages: () => api.get('/languages'),

  // Cosmetic rename of a diarized label (e.g. "Speaker_00") to a human name.
  // remember defaults to true — the voice is stored permanently for future meetings.
  renameSpeaker: (meetingId, speakerLabel, name, { remember = true, overwrite = false } = {}) => {
    const formData = new FormData()
    formData.append('name', name)
    formData.append('remember', remember ? 'true' : 'false')
    formData.append('overwrite', overwrite ? 'true' : 'false')
    return api.patch(
      `/meetings/${meetingId}/speakers/${encodeURIComponent(speakerLabel)}`,
      formData,
      remember ? { timeout: 0 } : undefined
    )
  },

  deleteMeetingSpeaker: (meetingId, speakerLabel) =>
    api.delete(`/meetings/${meetingId}/speakers/${encodeURIComponent(speakerLabel)}`),

  // Enroll a voice profile from meeting audio without renaming (or after a
  // prior rename). Uses the same ECAPA + speakers table path as /enroll.
  enrollSpeaker: (meetingId, speakerLabel, name, { overwrite = false } = {}) => {
    const formData = new FormData()
    formData.append('name', name)
    formData.append('overwrite', overwrite ? 'true' : 'false')
    return api.post(
      `/meetings/${meetingId}/speakers/${encodeURIComponent(speakerLabel)}/enroll`,
      formData,
      { timeout: 0 }
    )
  },

  // Re-match Speaker_XX labels against voices enrolled in Settings.
  identifySpeakers: (meetingId) =>
    api.post(`/meetings/${meetingId}/identify-speakers`, null, { timeout: 0 }),

  // The <audio> element fetches this itself, so it needs a plain URL rather
  // than an axios call.
  audioUrl: (id) => `${API_BASE_URL}/meetings/${id}/audio`,

  // Attach or replace audio on an existing meeting and restart transcription.
  uploadAudio: (meetingId, file, onUploadProgress) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/meetings/${meetingId}/audio`, formData, {
      timeout: 0,
      onUploadProgress
    })
  },
}

// Speaker endpoints live at the API root, not under /api.
const ROOT_URL = API_BASE_URL.replace(/\/api\/?$/, '')

export const speakersApi = {
  list: () => api.get('/speakers', { baseURL: ROOT_URL }),

  // `audio` is whatever MediaRecorder produced — webm/Opus in Chrome and
  // Firefox, mp4 in Safari. The backend decodes it with ffmpeg, so the
  // container doesn't matter, but the filename extension is sent along
  // because the decoder uses it to pick a demuxer.
  enroll: (name, blob) => {
    const ext = blob.type.includes('mp4') ? 'm4a' : 'webm'
    const formData = new FormData()
    formData.append('name', name)
    formData.append('audio', blob, `enrollment.${ext}`)
    return api.post('/enroll', formData, { baseURL: ROOT_URL, timeout: 0 })
  },

  remove: (name) =>
    api.delete(`/speakers/${encodeURIComponent(name)}`, { baseURL: ROOT_URL })
}
