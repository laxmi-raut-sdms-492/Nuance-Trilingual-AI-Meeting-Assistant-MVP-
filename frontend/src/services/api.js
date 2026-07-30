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

  search: (query) => api.get('/search', { params: { q: query } }),

  languages: () => api.get('/languages'),

  // Cosmetic rename of a diarized label (e.g. "Speaker_00") to a human name,
  // scoped to one meeting. Does not touch voice profiles -- see speakersApi
  // for enrolling a voice so future meetings auto-label it.
  renameSpeaker: (meetingId, speakerLabel, name) => {
    const formData = new FormData()
    formData.append('name', name)
    return api.patch(
      `/meetings/${meetingId}/speakers/${encodeURIComponent(speakerLabel)}`,
      formData
    )
  },

  // The <audio> element fetches this itself, so it needs a plain URL rather
  // than an axios call.
  audioUrl: (id) => `${API_BASE_URL}/meetings/${id}/audio`
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
