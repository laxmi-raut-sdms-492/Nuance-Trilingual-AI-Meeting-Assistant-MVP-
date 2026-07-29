import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  ListChecks,
  Trash2,
  BarChart3,
  Users,
  Sparkles,
  ChevronsLeft,
  ChevronsRight
} from 'lucide-react'
import { useState } from 'react'
import Avatar from '../common/Avatar.jsx'
import { useUser } from '../../context/UserContext.jsx'

const NavItem = ({ to, icon: Icon, label, collapsed }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 group ${
        isActive
          ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/40 dark:text-primary-300'
          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800 dark:hover:text-white'
      }`
    }
  >
    <Icon size={18} className="shrink-0" />
    {!collapsed && <span className="truncate">{label}</span>}
  </NavLink>
)

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const { profile } = useUser()
  const displayName = profile.name || 'Set up your profile'
  const displayEmail = profile.email || 'Add your email in Settings'

  return (
    <aside
      className={`h-screen sticky top-0 flex flex-col border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 transition-all duration-300 ${
        collapsed ? 'w-[76px]' : 'w-[260px]'
      }`}
    >
      <div className="flex items-center justify-between px-4 py-5">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-600 to-primary-500 flex items-center justify-center shrink-0">
            <Sparkles size={18} className="text-white" />
          </div>
          {!collapsed && (
            <div className="leading-tight">
              <p className="font-bold text-gray-900 dark:text-white text-[15px]">MeetIQ</p>
              <p className="text-[11px] text-gray-400 -mt-0.5">AI Meeting Intelligence</p>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1">
        <NavItem to="/dashboard" icon={LayoutDashboard} label="Dashboard" collapsed={collapsed} />
        <NavItem to="/meetings" icon={ListChecks} label="All Meetings" collapsed={collapsed} />
        <NavItem to="/meetings/trash" icon={Trash2} label="Trash" collapsed={collapsed} />
        <NavItem to="/analytics/insights" icon={BarChart3} label="Insights" collapsed={collapsed} />
        <NavItem to="/settings/members" icon={Users} label="Members" collapsed={collapsed} />
      </div>

      <NavLink
        to="/profile"
        className="border-t border-gray-100 dark:border-gray-800 p-3 flex items-center gap-3 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
      >
        <Avatar name={profile.name} size={36} />
        {!collapsed && (
          <div className="flex-1 overflow-hidden">
            <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">{displayName}</p>
            <p className="text-xs text-gray-400 truncate">{displayEmail}</p>
          </div>
        )}
        <button
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            setCollapsed((c) => !c)
          }}
          className="ml-auto text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
        >
          {collapsed ? <ChevronsRight size={16} /> : <ChevronsLeft size={16} />}
        </button>
      </NavLink>
    </aside>
  )
}
