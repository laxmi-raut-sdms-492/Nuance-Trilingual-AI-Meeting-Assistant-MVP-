import { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import Icon from '../../components/common/Icon.jsx'
import Avatar from '../../components/common/Avatar.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import SpeakerEnrollment from './SpeakerEnrollment.jsx'
import { useUser } from '../../context/UserContext.jsx'
import { useMembers } from '../../context/MembersContext.jsx'
import { useTheme } from '../../context/ThemeContext.jsx'

/** Ported from the design export (settings_members_prefs). */

const TITLES = {
  preferences: 'Preferences',
  members: 'Members',
  profile: 'Profile',
  speakers: 'Speaker Enrollment',
}

const TABS = [
  { to: '/settings/members', key: 'members', label: 'Members' },
  { to: '/settings/speakers', key: 'speakers', label: 'Speakers' },
  { to: '/settings/preferences', key: 'preferences', label: 'Preferences' },
  { to: '/profile', key: 'profile', label: 'Profile' },
]

const panel = 'bg-surface border border-border rounded-xl p-6'
const input =
  'input-base w-full px-4 py-2.5 rounded-lg border font-transcript-body text-transcript-body placeholder:text-text-faint'

export default function Settings({ tab }) {
  return (
    <>
      <div>
        <p className="font-meta-data text-meta-data text-text-muted mb-1">Settings</p>
        <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
          {TITLES[tab] || 'Settings'}
        </h2>
      </div>

      <div className="flex items-center gap-6 border-b border-border px-2 overflow-x-auto hide-scrollbar">
        {TABS.map((t) => (
          <NavLink
            key={t.key}
            to={t.to}
            className={`pb-3 border-b-2 font-label-sm text-label-sm uppercase tracking-widest transition-colors whitespace-nowrap ${
              tab === t.key
                ? 'border-primary-container text-primary'
                : 'border-transparent text-text-muted hover:text-text-primary'
            }`}
          >
            {t.label}
          </NavLink>
        ))}
      </div>

      <div className="max-w-3xl w-full">
        {tab === 'preferences' && <PreferencesTab />}
        {tab === 'members' && <MembersTab />}
        {tab === 'profile' && <ProfileTab />}
        {tab === 'speakers' && <SpeakerEnrollment />}
      </div>
    </>
  )
}

function PreferencesTab() {
  const { dark, toggleDark } = useTheme()

  return (
    <div className={panel}>
      <Toggle
        label="Dark Mode"
        description="Nuance is designed dark. The light palette is derived from it."
        checked={dark}
        onChange={toggleDark}
      />
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
    <div className={panel}>
      <div className="flex items-center gap-4 mb-6">
        <Avatar name={name} size={56} />
        <div>
          <p className="text-text-primary">Your profile</p>
          <p className="font-meta-data text-meta-data text-text-muted">
            Your avatar is generated from your initials.
          </p>
        </div>
      </div>
      <form onSubmit={save} className="flex flex-col gap-4 max-w-sm">
        <Field label="Full Name" value={name} onChange={setName} placeholder="e.g. Priya Sharma" required />
        <Field
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          placeholder="you@company.com"
          required
        />
        <div className="flex items-center gap-3">
          <button
            type="submit"
            className="px-5 py-2.5 rounded-lg bg-cta text-on-cta font-label-sm text-label-sm hover:bg-primary-container transition-colors"
          >
            Save Changes
          </button>
          {saved && (
            <span className="font-meta-data text-meta-data text-success flex items-center gap-1">
              <Icon name="check_circle" size={16} /> Saved
            </span>
          )}
        </div>
      </form>
    </div>
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
    <div className="flex flex-col gap-6">
      <div className={panel}>
        <h3 className="font-sidebar-header text-sidebar-header text-text-primary mb-4">
          Add a Member
        </h3>
        <form onSubmit={submit} className="flex flex-col sm:flex-row gap-3">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Full name"
            className={input}
          />
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            type="email"
            placeholder="Email"
            className={input}
          />
          <button
            type="submit"
            className="px-5 py-2.5 rounded-lg bg-cta text-on-cta font-label-sm text-label-sm hover:bg-primary-container transition-colors shrink-0 flex items-center gap-2 justify-center"
          >
            <Icon name="person_add" size={18} />
            Add
          </button>
        </form>
        <p className="font-meta-data text-meta-data text-text-muted mt-3">
          Members are stored in this browser. They are a roster for the dashboard count, not
          accounts — there is no auth system.
        </p>
      </div>

      <div className={panel}>
        <h3 className="font-sidebar-header text-sidebar-header text-text-primary mb-4">
          Team Members
        </h3>
        {members.length === 0 ? (
          <EmptyState
            icon="group"
            title="No members yet"
            subtitle="Set up your profile to appear here, then add teammates above."
          />
        ) : (
          <div className="flex flex-col gap-3">
            {members.map((m) => (
              <div
                key={m.id}
                className="flex items-center gap-3 p-4 rounded-lg border border-border bg-surface-raised"
              >
                <Avatar name={m.name} size={40} />
                <div className="flex-1 min-w-0">
                  <span className="text-text-primary block truncate">{m.name}</span>
                  <span className="font-meta-data text-meta-data text-text-muted truncate block">
                    {m.email}
                  </span>
                </div>
                <span className="font-label-sm text-label-sm text-text-faint uppercase tracking-wider shrink-0">
                  {m.role}
                </span>
                {m.id !== 'self' && (
                  <button
                    type="button"
                    onClick={() => removeMember(m.id)}
                    aria-label={`Remove ${m.name}`}
                    className="text-text-muted hover:text-error shrink-0 p-1 transition-colors"
                  >
                    <Icon name="delete" className="text-[18px]" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text', placeholder, required }) {
  return (
    <label className="block">
      <span className="font-label-sm text-label-sm text-text-muted mb-2 block uppercase tracking-wider">
        {label}
      </span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required={required}
        className={input}
      />
    </label>
  )
}

function Toggle({ label, description, checked, onChange }) {
  return (
    <div className="flex items-center justify-between gap-4 py-4 first:pt-0 last:pb-0">
      <div className="min-w-0">
        <p className="text-text-primary">{label}</p>
        {description && (
          <p className="font-meta-data text-meta-data text-text-muted mt-1">{description}</p>
        )}
      </div>
      {/* The knob is absolutely positioned and MUST set `left`. With only
          `top`, it falls back to its static position — which a <button>
          centres — and translating from there threw it outside the track.
          The border also lives on both states so toggling doesn't shift the
          box by a pixel. */}
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={`w-11 h-6 rounded-full transition-colors relative shrink-0 border focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-container ${
          checked ? 'bg-cta border-cta' : 'bg-surface-container-high border-border'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-[18px] h-[18px] rounded-full shadow transition-transform duration-150 ${
            checked ? 'translate-x-[20px] bg-on-cta' : 'translate-x-0 bg-text-muted'
          }`}
        />
      </button>
    </div>
  )
}
