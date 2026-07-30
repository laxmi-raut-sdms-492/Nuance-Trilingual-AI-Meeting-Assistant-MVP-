import { NavLink, useNavigate } from 'react-router-dom'
import Icon from '../common/Icon.jsx'

/**
 * Desktop rail, ported from the design export (every screen carries an
 * identical copy of this nav). Hidden below md: — mobile navigates via the
 * bottom tab bar in MobileNav.jsx, which is what the export uses instead of
 * the drawer the implementation brief assumed.
 *
 * Nav order is the export's, which differs from the old build: Trash moved
 * below Members, and Profile Settings is pinned to the bottom.
 */

const NAV = [
  { to: '/dashboard', icon: 'dashboard', label: 'Dashboard' },
  { to: '/meetings', icon: 'event_note', label: 'All Meetings', end: true },
  { to: '/analytics/insights', icon: 'insights', label: 'Insights' },
  { to: '/settings/members', icon: 'group', label: 'Members' },
  { to: '/meetings/trash', icon: 'delete', label: 'Trash' },
]

function NavItem({ to, icon, label, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
          isActive
            ? 'bg-surface-raised text-primary font-bold'
            : 'text-text-muted font-normal hover:bg-surface-raised'
        }`
      }
    >
      {({ isActive }) => (
        <>
          <Icon name={icon} filled={isActive} />
          {label}
        </>
      )}
    </NavLink>
  )
}

export default function Sidebar() {
  const navigate = useNavigate()

  return (
    <nav className="w-sidebar-width h-screen fixed left-0 top-0 hidden md:flex flex-col bg-surface border-r border-border p-4 gap-6 z-50">
      <div className="flex items-center gap-3 px-2 mb-4">
        <div className="w-8 h-8 rounded bg-primary-container flex items-center justify-center text-on-primary-container">
          <Icon name="graphic_eq" filled />
        </div>
        <div>
          <h1 className="font-sidebar-header text-sidebar-header text-text-primary">Nuance</h1>
          <p className="font-label-sm text-label-sm text-text-muted">AI Intelligence Tool</p>
        </div>
      </div>

      <button
        type="button"
        onClick={() => navigate('/upload')}
        className="w-full bg-cta hover:bg-primary-container text-on-cta font-label-sm text-label-sm py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors duration-150 ease-in-out scale-95 hover:scale-100"
      >
        <Icon name="add" />
        New Meeting
      </button>

      <div className="flex-1 flex flex-col gap-1 mt-4">
        {NAV.map((item) => (
          <NavItem key={item.to} {...item} />
        ))}
      </div>

      <div className="mt-auto">
        <NavItem to="/profile" icon="settings" label="Profile Settings" />
      </div>
    </nav>
  )
}
