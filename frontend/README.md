# MeetIQ — AI Meeting Intelligence (Frontend)

A production-ready React frontend for the **Nuance** AI Meeting Intelligence system, built with Vite, Tailwind CSS, React Router, Recharts, and Framer Motion.

**This build uses real, accessible data — not hardcoded mock content.** When you upload a file on the Upload page, it's saved as an actual meeting record (real file name, real size, real upload date/time) in your browser's `localStorage` via `MeetingsContext`. Every screen — Dashboard, All Meetings, Meeting Details, Analytics — reads from that same live store, so what you see is always what you've actually uploaded. Charts are computed from your real upload history. Fields that require the transcription/summarization pipeline (transcript, AI summary, action items, speaker breakdown, keywords) are left empty with an honest "not generated yet" state until the FSD backend is connected — nothing is faked.

## Tech Stack
- React 18 + Vite
- Tailwind CSS
- React Router DOM
- Axios
- Recharts (analytics charts)
- Lucide React + React Icons
- React Hot Toast
- Framer Motion

## Folder Structure
```
src/
  assets/
  components/
    common/       Button, Card, Badge, Loader, EmptyState, Pagination
    layout/       Sidebar, Navbar, MainLayout
    cards/        StatCard
    charts/       WeeklyChart, TimelineChart, SpeakerPie
  pages/
    Dashboard/
    Meetings/
    MeetingDetails/
    UploadMeeting/
    Analytics/
    Settings/
  context/        MeetingsContext.jsx — the single source of truth for meeting data
  data/           team.js — static team roster (org data, not fake meetings)
  services/       api.js (axios client + endpoint wrappers for the real backend)
  utils/          formatters.js
  hooks/
  routes/
  styles/
  App.jsx
  main.jsx
```

## How data flows
1. **Upload page** (`/upload`) — pick or drag a real `.mp3`/`.wav`/`.mp4`/`.m4a` file.
2. `MeetingsContext.addMeeting(file)` creates a record with the file's real name, size, type, and timestamp, and saves it to `localStorage` (key: `meetiq:meetings`).
3. **Dashboard**, **All Meetings**, and **Analytics** all read from `useMeetings()` — so a file you upload shows up everywhere instantly, and persists across page refreshes.
4. **Meeting Details** shows the real file metadata immediately. Transcript / Summary / Insights / Speakers tabs show an empty state until those fields are filled in by a real backend call (see below).
5. Deleting a meeting (trash icon in All Meetings) removes it from the store everywhere.

## Requirements
- Node.js **18+** (Node 20 LTS recommended)
- npm 9+

Check your versions:
```bash
node -v
npm -v
```

## Setup — Step by Step

### 1. Unzip the project
Unzip `meetiq-frontend.zip` and open a terminal in the extracted folder:
```bash
cd meetiq-frontend
```

### 2. Install dependencies
```bash
npm install
```
This installs React, Vite, Tailwind, Recharts, and all other packages listed in `package.json`.

### 3. Configure environment variables (optional)
Copy the example env file:
```bash
cp .env.example .env
```
Edit `.env` when you connect a real backend:
```
VITE_API_BASE_URL=http://localhost:8000/api
```
The app works fine without this step — it stores real upload metadata locally until a backend is wired up.

### 4. Start the development server
```bash
npm run dev
```
Vite will start the dev server and print a local URL, typically:
```
  VITE v5.x.x  ready in 400 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```
Open **http://localhost:5173** in your browser.

### 5. Build for production
```bash
npm run build
```
This creates an optimized production bundle in the `dist/` folder.

### 6. Preview the production build locally
```bash
npm run preview
```
This serves the `dist/` build, typically at **http://localhost:4173**.

## Connecting the Real Backend
The FSD describes a Python backend with these endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/meetings` | POST | Upload audio, trigger transcription/summarization pipeline |
| `/api/meetings` | GET | List all meetings |
| `/api/meetings/{id}` | GET | Fetch a single meeting's full transcript/summary |
| `/api/search` | GET | Full-text search across transcripts/summaries |

`src/services/api.js` already contains an Axios client (`meetingsApi`) pointed at `VITE_API_BASE_URL`. To go live:
1. Deploy the backend pipeline described in the FSD (Replicate + OpenRouter + PostgreSQL).
2. Set `VITE_API_BASE_URL` in `.env` to your backend's base URL.
3. In `src/pages/UploadMeeting/UploadMeeting.jsx`, replace the local `addMeeting(file)` call with `meetingsApi.create(formData)`, uploading the real file to your backend instead of (or in addition to) `localStorage`.
4. In `src/context/MeetingsContext.jsx`, replace the `localStorage`-backed state with data fetched from `meetingsApi.list()` / `meetingsApi.getById(id)`, so transcript, summary, action items, and speaker stats come back populated from the real pipeline instead of showing "not generated yet".

## Notes
- Nothing on screen is placeholder/mock meeting content — every meeting you see was actually uploaded by you, in this browser.
- `src/data/team.js` is a static team roster (used in Settings → Members and Analytics) since there's no user-management backend yet; it's organizational config, not fabricated meeting data.
- Dark mode toggle in the navbar is a UI placeholder — wire it to a theme context if needed.
- Tailwind color palette and shadows are defined in `tailwind.config.js` under `theme.extend`.

