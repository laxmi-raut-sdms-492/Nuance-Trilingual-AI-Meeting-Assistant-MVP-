import { useState, useEffect } from 'react'
import { UserPlus, Trash2 } from 'lucide-react'
import Card from '../../components/common/Card.jsx'
import Button from '../../components/common/Button.jsx'
import Avatar from '../../components/common/Avatar.jsx'
import { useUser } from '../../context/UserContext.jsx'
import { useMembers } from '../../context/MembersContext.jsx'
import { useTheme } from '../../context/ThemeContext.jsx'

const titles = {
  integrations: 'Integrations',
  preferences: 'Preferences',
  members: 'Members',
  profile: 'Profile'
}

const INTEGRATION_APPS = ['Google Calendar', 'Slack', 'Zoom', 'Microsoft Teams']
const CONNECTIONS_KEY = 'meetiq:integrations'
const NOTIFS_KEY = 'meetiq:notifications'

function loadConnections() {
  try {
    const raw = localStorage.getItem(CONNECTIONS_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function loadNotifPref() {
  try {
    const raw = localStorage.getItem(NOTIFS_KEY)
    return raw === null ? true : raw === 'true'
  } catch {
    return true
  }
}

export default function Settings({ tab }) {
  const { dark, toggleDark } = useTheme()
  const [notifs, setNotifs] = useState(() => loadNotifPref())
  const [connections, setConnections] = useState(() => loadConnections())

  useEffect(() => {
    localStorage.setItem(NOTIFS_KEY, String(notifs))
  }, [notifs])

  useEffect(() => {
    localStorage.setItem(CONNECTIONS_KEY, JSON.stringify(connections))
  }, [connections])

  const toggleConnection = (app) => {
    setConnections((prev) => ({ ...prev, [app]: !prev[app] }))
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{titles[tab] || 'Settings'}</h1>

      {tab === 'integrations' && (
        <Card>
          <h2 className="font-semibold text-gray-900 dark:text-white mb-1">Connected Apps</h2>
          <p className="text-xs text-gray-400 mb-4">
            This toggles a saved connection state — real OAuth sign-in will replace this once a
            backend is wired up, but your choice persists here for now.
          </p>
          <div className="space-y-3">
            {INTEGRATION_APPS.map((app) => (
              <div key={app} className="flex items-center justify-between p-3 rounded-xl border border-gray-100 dark:border-gray-700">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-200">{app}</span>
                <Button
                  variant={connections[app] ? 'secondary' : 'primary'}
                  className="!px-3 !py-1.5 text-xs"
                  onClick={() => toggleConnection(app)}
                >
                  {connections[app] ? 'Disconnect' : 'Connect'}
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      {tab === 'preferences' && (
        <Card className="space-y-4">
          <Toggle label="Dark Mode" checked={dark} onChange={toggleDark} />
          <Toggle label="Email Notifications" checked={notifs} onChange={setNotifs} />
        </Card>
      )}

      {tab === 'members' && <MembersTab />}

      {tab === 'profile' && <ProfileTab />}
    </div>
  )
}

function ProfileTab() {
  const { profile, updateProfile } = useUser()
  const [name, setName] = useState(profile.name)
  const [email, setEmail] = useState(profile.email)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setName(profile.name)
    setEmail(profile.email)
  }, [profile.name, profile.email])

  const save = (e) => {
    e.preventDefault()
    updateProfile({ name: name.trim(), email: email.trim() })
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <Card>
      <div className="flex items-center gap-4 mb-6">
        <Avatar name={name} size={56} />
        <div>
          <p className="text-sm font-semibold text-gray-900 dark:text-white">Your profile</p>
          <p className="text-xs text-gray-400">Your avatar is generated from your initials.</p>
        </div>
      </div>
      <form onSubmit={save} className="space-y-4 max-w-sm">
        <Field label="Full Name" value={name} onChange={setName} placeholder="e.g. Priya Sharma" required />
        <Field label="Email" type="email" value={email} onChange={setEmail} placeholder="you@company.com" required />
        <div className="flex items-center gap-3">
          <Button type="submit">Save Changes</Button>
          {saved && <span className="text-xs text-green-600 font-medium">Saved</span>}
        </div>
      </form>
    </Card>
  )
}

function MembersTab() {
  const { members, addMember, removeMember } = useMembers()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')

  const submit = (e) => {
    e.preventDefault()
    if (!name.trim() || !email.trim()) return
    addMember({ name: name.trim(), email: email.trim() })
    setName('')
    setEmail('')
  }

  return (
    <div className="space-y-5">
      <Card>
        <h2 className="font-semibold text-gray-900 dark:text-white mb-4">Add a Member</h2>
        <form onSubmit={submit} className="flex flex-col sm:flex-row gap-3">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Full name"
            className="flex-1 px-3 py-2 rounded-xl border border-gray-200 dark:border-gray-700 dark:bg-gray-800 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-200"
          />
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            type="email"
            placeholder="Email"
            className="flex-1 px-3 py-2 rounded-xl border border-gray-200 dark:border-gray-700 dark:bg-gray-800 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-200"
          />
          <Button type="submit" icon={UserPlus} className="shrink-0">
            Add
          </Button>
        </form>
      </Card>

      <Card>
        <h2 className="font-semibold text-gray-900 dark:text-white mb-4">Team Members</h2>
        {members.length === 0 ? (
          <p className="text-sm text-gray-400">
            No members yet — set up your profile to appear here, then invite teammates above.
          </p>
        ) : (
          <div className="space-y-3">
            {members.map((m) => (
              <div key={m.id} className="flex items-center gap-3 p-3 rounded-xl border border-gray-100 dark:border-gray-700">
                <Avatar name={m.name} size={36} />
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-200 block truncate">{m.name}</span>
                  <span className="text-xs text-gray-400 truncate block">{m.email}</span>
                </div>
                <span className="text-xs text-gray-400 shrink-0">{m.role}</span>
                {m.id !== 'self' && (
                  <button
                    onClick={() => removeMember(m.id)}
                    className="text-gray-300 hover:text-red-500 shrink-0"
                    title="Remove member"
                  >
                    <Trash2 size={15} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text', placeholder, required }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-gray-500 mb-1 block">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required={required}
        className="w-full px-3 py-2 rounded-xl border border-gray-200 dark:border-gray-700 dark:bg-gray-800 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-200"
      />
    </label>
  )
}

function Toggle({ label, checked, onChange }) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-200">{label}</span>
      <button
        onClick={() => onChange(!checked)}
        className={`w-11 h-6 rounded-full transition-colors relative ${checked ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-700'}`}
      >
        <span
          className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
            checked ? 'translate-x-5' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  )
}
