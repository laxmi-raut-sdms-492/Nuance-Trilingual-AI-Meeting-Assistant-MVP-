import { Search, Bell, Moon, Sun } from 'lucide-react'
import { useState, useMemo } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import Avatar from '../common/Avatar.jsx'
import { useUser } from '../../context/UserContext.jsx'
import { useTheme } from '../../context/ThemeContext.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

export default function Navbar() {
  const navigate = useNavigate()
  const { dark, toggleDark } = useTheme()
  const { profile } = useUser()
  const { meetings } = useMeetings()
  const [query, setQuery] = useState('')
  const [notifOpen, setNotifOpen] = useState(false)

  // Real, derived from actual meeting state - not a static always-on badge.
  const processing = useMemo(() => meetings.filter((m) => m.status === 'Processing'), [meetings])

  const runSearch = (e) => {
    e.preventDefault()
    const q = query.trim()
    navigate(q ? `/meetings?q=${encodeURIComponent(q)}` : '/meetings')
  }

  return (
    <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur border-b border-gray-200 dark:border-gray-800 px-6 py-3 flex items-center justify-between gap-4">
      <form onSubmit={runSearch} className="relative flex-1 max-w-xl">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search meetings, keywords, people..."
          className="w-full pl-9 pr-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-200 focus:border-primary-400 transition-all"
        />
      </form>

      <div className="flex items-center gap-3">
        <button
          onClick={toggleDark}
          className="w-9 h-9 rounded-full flex items-center justify-center text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {dark ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        <div className="relative">
          <button
            onClick={() => setNotifOpen((o) => !o)}
            className="relative w-9 h-9 rounded-full flex items-center justify-center text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <Bell size={18} />
            {processing.length > 0 && (
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-red-500 ring-2 ring-white dark:ring-gray-900" />
            )}
          </button>

          {notifOpen && (
            <div className="absolute right-0 mt-2 w-72 rounded-xl border border-gray-100 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-soft p-3 z-20">
              <p className="text-xs font-semibold text-gray-400 mb-2">Notifications</p>
              {processing.length === 0 ? (
                <p className="text-sm text-gray-400 py-4 text-center">You're all caught up.</p>
              ) : (
                <div className="space-y-1">
                  {processing.slice(0, 5).map((m) => (
                    <Link
                      key={m.id}
                      to={`/meetings/${m.id}`}
                      onClick={() => setNotifOpen(false)}
                      className="block px-2 py-2 rounded-lg text-sm hover:bg-gray-50 dark:hover:bg-gray-700"
                    >
                      <span className="font-medium text-gray-800 dark:text-gray-100">{m.title}</span>
                      <span className="block text-xs text-gray-400">Still processing</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <Link to="/profile" className="flex items-center gap-2 pl-2 border-l border-gray-200 dark:border-gray-700">
          <Avatar name={profile.name} size={36} />
        </Link>
      </div>
    </header>
  )
}
