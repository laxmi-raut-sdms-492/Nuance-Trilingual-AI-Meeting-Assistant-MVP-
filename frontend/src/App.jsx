import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './components/layout/MainLayout.jsx'
import { MeetingsProvider } from './context/MeetingsContext.jsx'
import { UserProvider } from './context/UserContext.jsx'
import { MembersProvider } from './context/MembersContext.jsx'
import { ThemeProvider } from './context/ThemeContext.jsx'
import Dashboard from './pages/Dashboard/Dashboard.jsx'
import MeetingList from './pages/Meetings/MeetingList.jsx'
import MeetingDetails from './pages/MeetingDetails/MeetingDetails.jsx'
import UploadMeeting from './pages/UploadMeeting/UploadMeeting.jsx'
import AnalyticsPage from './pages/Analytics/AnalyticsPage.jsx'
import Settings from './pages/Settings/Settings.jsx'
import NotFound from './pages/NotFound.jsx'

export default function App() {
  return (
    <ThemeProvider>
      <UserProvider>
        <MembersProvider>
          <MeetingsProvider>
            <BrowserRouter>
              <Routes>
                <Route element={<MainLayout />}>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />

                  <Route path="/meetings" element={<MeetingList />} />
                  <Route path="/meetings/trash" element={<MeetingList filter="trash" />} />
                  <Route path="/meetings/:id" element={<MeetingDetails />} />

                  <Route path="/upload" element={<UploadMeeting />} />

                  <Route path="/analytics/insights" element={<AnalyticsPage tab="insights" />} />

                  <Route path="/settings/members" element={<Settings tab="members" />} />
                  <Route path="/members" element={<Settings tab="members" />} />
                  <Route path="/settings/speakers" element={<Settings tab="speakers" />} />
                  <Route path="/settings/preferences" element={<Settings tab="preferences" />} />
                  <Route path="/profile" element={<Settings tab="profile" />} />

                  <Route path="*" element={<NotFound />} />
                </Route>
              </Routes>
            </BrowserRouter>
          </MeetingsProvider>
        </MembersProvider>
      </UserProvider>
    </ThemeProvider>
  )
}
