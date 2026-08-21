import { useState, useMemo } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import Icon from '../common/Icon.jsx'
import Avatar from '../common/Avatar.jsx'
import { useUser } from '../../context/UserContext.jsx'
import { useTheme } from '../../context/ThemeContext.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

/**
 * Top app bar, ported from the design export.
 *
 * The export shows a brand lockup on mobile, a spacer on desktop, then
 * contrast / notifications / avatar on the right. It has no search field —
 * the old build's search is kept below md:hidden on the left because it is
 * wired to real behaviour and removing it would lose a working feature; on
 * mobile the export's brand lockup takes that slot.
 */
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
    <header className="sticky top-0 w-full z-40 bg-background/80 backdrop-blur-md border-b border-border flex justify-between items-center px-gutter py-4 gap-4">
      {/* Brand lockup — mobile only, per the export */}
      <div className="md:hidden flex items-center gap-2 shrink-0">
        <div className="w-6 h-6 rounded bg-primary-container flex items-center justify-center text-on-primary-container">
          <Icon name="graphic_eq" filled size={16} />
        </div>
        <h1 className="font-sidebar-header text-sidebar-header text-text-primary">Nuance</h1>
      </div>

      <form onSubmit={runSearch} className="relative flex-1 max-w-xl hidden md:block">
        <Icon
          name="search"
          size={18}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
        />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search meetings, topics, people..."
          className="w-full pl-10 pr-4 py-2 rounded-lg border border-border bg-surface-raised text-text-primary placeholder:text-text-faint font-meta-data text-meta-data focus:outline-none focus:border-primary-container transition-colors"
        />
      </form>

      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={toggleDark}
          className="text-text-muted hover:text-primary transition-colors opacity-80 hover:opacity-100 p-2 rounded-full hover:bg-surface-raised"
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          <Icon name="contrast" />
        </button>

        <div className="relative">
          <button
            type="button"
            onClick={() => setNotifOpen((o) => !o)}
            className="text-text-muted hover:text-primary transition-colors opacity-80 hover:opacity-100 p-2 rounded-full hover:bg-surface-raised relative"
          >
            <Icon name="notifications" />
            {processing.length > 0 && (
              <span className="absolute top-2 right-2 w-2 h-2 bg-primary-container rounded-full" />
            )}
          </button>

          {notifOpen && (
            <div className="absolute right-0 mt-2 w-72 rounded-xl border border-border bg-surface-raised shadow-lg p-3 z-20">
              <p className="font-label-sm text-label-sm text-text-muted uppercase tracking-wider mb-2">
                Notifications
              </p>
              {processing.length === 0 ? (
                <p className="font-meta-data text-meta-data text-text-muted py-4 text-center">
                  You're all caught up.
                </p>
              ) : (
                <div className="space-y-1">
                  {processing.slice(0, 5).map((m) => (
                    <Link
                      key={m.id}
                      to={`/meetings/${m.id}`}
                      onClick={() => setNotifOpen(false)}
                      className="block px-2 py-2 rounded-lg hover:bg-surface-container-high transition-colors"
                    >
                      <span className="text-text-primary">{m.title}</span>
                      <span className="block font-meta-data text-meta-data text-processing">
                        Still processing
                      </span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <Link
          to="/profile"
          className="w-8 h-8 rounded-full bg-surface-raised border border-border overflow-hidden cursor-pointer hover:border-primary-container transition-colors flex items-center justify-center"
        >
          <Avatar name={profile.name} size={32} />
        </Link>
      </div>
    </header>
  )
}
