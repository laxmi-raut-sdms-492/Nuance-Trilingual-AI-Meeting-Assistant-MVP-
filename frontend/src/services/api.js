import axios from 'axios'

// Base API client - swap VITE_API_BASE_URL in .env to point to your real backend.
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  timeout: 15000
})

// Mirrors the FSD endpoints. Currently unused because meeting data lives in
// this browser (localStorage/IndexedDB via MeetingsContext) until the
// Python backend from the FSD is deployed and wired up here.
export const meetingsApi = {
  list: () => api.get('/meetings'),
  create: (formData) =>
    api.post('/meetings', formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  getById: (id) => api.get(`/meetings/${id}`),
  search: (query) => api.get('/search', { params: { q: query } })
}
