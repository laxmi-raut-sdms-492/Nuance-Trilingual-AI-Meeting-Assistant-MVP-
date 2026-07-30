import { NavLink, useNavigate } from 'react-router-dom'
import Icon from '../common/Icon.jsx'

/**
 * Mobile bottom tab bar, ported from the design export.
 *
 * The implementation brief specified a hamburger drawer here; the export does
 * not have one. Nine of its screens carry this fixed bottom bar instead
 * (`md:hidden`, h-16), with the Add action raised above the bar on the centre
 * tab. The export is the design of record, so this follows it.
 *
 * MainLayout pays for this with pb-20 on mobile so content clears the bar.
 */

const TABS = [
  { to: '/dashboard', icon: 'home', label: 'Home' },
  { to: '/meetings', icon: 'list_alt', label: 'Meetings', end: true },
  { to: '/upload', icon: 'add', label: 'Add', raised: true },
  { to: '/analytics/insights', icon: 'analytics', label: 'Insights' },
  { to: '/settings/preferences', icon: 'menu', label: 'More' },
]

export default function MobileNav() {
  const navigate = useNavigate()

  return (
    <nav className="fixed bottom-0 w-full md:hidden z-50 bg-surface border-t border-border shadow-lg flex justify-around items-center h-16 px-2">
      {TABS.map(({ to, icon, label, end, raised }) =>
        raised ? (
          // The Add tab sits proud of the bar and is a button, not a nav
          // link, so it reads as the primary action rather than a location.
          <button
            key={to}
            type="button"
            onClick={() => navigate(to)}
            className="flex flex-col items-center justify-center text-text-muted p-2 active:bg-surface-container-high rounded-xl transition-colors relative -top-3"
          >
            <div className="bg-cta text-on-cta rounded-full p-2 shadow-lg">
              <Icon name={icon} />
            </div>
            <span className="font-label-sm text-label-sm-mobile mt-1">{label}</span>
          </button>
        ) : (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex flex-col items-center justify-center p-2 rounded-xl active:bg-surface-container-high transition-colors ${
                isActive
                  ? 'bg-secondary-container text-on-secondary-container'
                  : 'text-text-muted'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon name={icon} filled={isActive} />
                <span className="font-label-sm text-label-sm-mobile">{label}</span>
              </>
            )}
          </NavLink>
        )
      )}
    </nav>
  )
}
