# MeetIQ — Nuance Trilingual Meeting Assistant (Frontend)

React frontend for the Nuance meeting assistant. Built with Vite, Tailwind CSS,
React Router, Recharts, and Framer Motion.

**The backend is the source of truth.** Uploads go to the Python API, which
stores the recording and transcribes it; every screen reads meetings back from
that API. Nothing is stored in the browser and nothing on screen is mock data.
Fields that need work not yet built (AI summary, decisions, action items,
keywords) stay empty with an honest "not generated yet" state rather than being
faked.

## Requirements

- Node.js **18+** (Node 20 LTS recommended)
- The backend running — see the root `README.md`. Without it, screens show a
  "cannot reach the API" error instead of silently falling back to fake data.

## Setup

```bash
npm install
cp .env.example .env      # VITE_API_BASE_URL=http://localhost:8000/api
npm run dev               # http://localhost:5173
```

Production build and preview:

```bash
npm run build             # -> dist/
npm run preview           # http://localhost:4173
```

Both 5173 and 4173 are in the backend's default `CORS_ORIGINS`.

## How data flows

1. **Create Meeting** (`/upload`) — enter a title and agenda, then upload a
   file or record live in the browser.
2. `MeetingsContext.addMeeting()` POSTs the real file to `/api/meetings` as
   multipart form data. The progress bar is the actual bytes on the wire
   reported by axios.
3. The backend responds immediately with a record in status `Processing` and
   transcribes in the background.
4. `MeetingsContext` polls `/api/meetings` every 3 seconds **only while
   something is still processing**, so status and progress advance on their own.
   Meeting Details polls its own record the same way.
5. Once `Completed`, the transcript renders with speaker, timestamp, and the
   **language detected for that individual line** — a meeting that switches
   between English, Hindi, and Marathi is labeled line by line.
6. Deleting a meeting removes the record, its transcript, and its stored audio
   on the server.

## Structure

```
src/
  components/
    common/       Button, Card, Badge, Loader, EmptyState, Pagination,
                  AudioPlayer (streams from the API), AudioRecorder
    layout/       Sidebar, Navbar, MainLayout
    cards/        StatCard
    charts/       WeeklyChart, TimelineChart, SpeakerPie
  pages/          Dashboard, Meetings, MeetingDetails, UploadMeeting,
                  Analytics, Settings
  context/        MeetingsContext  — fetch/poll/upload/delete against the API
                  MembersContext, UserContext, ThemeContext
  data/           team.js — static team roster (org config, not meeting data)
  services/       api.js — the single place the backend URL lives
  utils/          formatters.js
```

## Notes

- `src/services/api.js` is the only module that knows the backend URL. Point
  `VITE_API_BASE_URL` elsewhere and the whole app follows.
- Audio streams from `/api/meetings/{id}/audio` rather than browser storage, so
  a recording plays back on any machine, not only the one that uploaded it.
- `src/data/team.js` is a static roster used in Settings → Members, because
  there is no user-management backend yet. It is organizational config, not
  fabricated meeting data.
- Tailwind colours and shadows are defined in `tailwind.config.js` under
  `theme.extend`.
