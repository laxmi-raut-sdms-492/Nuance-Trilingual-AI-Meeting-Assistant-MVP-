import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './components/layout/MainLayout.jsx'
import { MeetingsProvider } from './context/MeetingsContext.jsx'
import { UserProvider } from './context/UserContext.jsx'
import { MembersProvider } from './context/MembersContext.jsx'
import { ThemeProvider } from './context/ThemeContext.jsx'
import { AuthProvider, useAuth } from './context/AuthContext.jsx'
import Login from './pages/Login/Login.jsx'
import Loader from './components/common/Loader.jsx'
import Dashboard from './pages/Dashboard/Dashboard.jsx'
import MeetingList from './pages/Meetings/MeetingList.jsx'
import MeetingDetails from './pages/MeetingDetails/MeetingDetails.jsx'
import UploadMeeting from './pages/UploadMeeting/UploadMeeting.jsx'
import AnalyticsPage from './pages/Analytics/AnalyticsPage.jsx'
import Settings from './pages/Settings/Settings.jsx'
import NotFound from './pages/NotFound.jsx'

/**
 * Nothing behind the password mounts until it is given: the data providers
 * sit inside this gate, so their first fetch happens signed in rather than
 * failing with 401 and showing empty lists.
 */
function AuthGate({ children }) {
  const { required, signedIn } = useAuth()
  if (required === null) return <Loader label="Connecting..." />
  if (!signedIn) return <Login />
  return children
}

export default function App() {
  return (
    <ThemeProvider>
      <UserProvider>
        <AuthProvider>
        <AuthGate>
        <MembersProvider>
          <MeetingsProvider>
            <BrowserRouter>
              <Routes>
                <Route element={<MainLayout />}>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />

                  {/* Keyed so page/filter state resets instead of leaking into Trash. */}
                  <Route path="/meetings" element={<MeetingList key="all" />} />
                  <Route path="/meetings/trash" element={<MeetingList key="trash" filter="trash" />} />
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
        </AuthGate>
        </AuthProvider>
      </UserProvider>
    </ThemeProvider>
  )
}
