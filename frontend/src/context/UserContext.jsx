import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const UserContext = createContext(null)
const STORAGE_KEY = 'meetiq:profile'

const DEFAULT_PROFILE = {
  name: '',
  email: '',
  role: 'Admin'
}

function loadProfile() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...DEFAULT_PROFILE, ...JSON.parse(raw) } : DEFAULT_PROFILE
  } catch {
    return DEFAULT_PROFILE
  }
}

export function UserProvider({ children }) {
  const [profile, setProfile] = useState(() => loadProfile())

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(profile))
    } catch {
      // ignore write failures
    }
  }, [profile])

  const updateProfile = useCallback((updates) => {
    setProfile((prev) => ({ ...prev, ...updates }))
  }, [])

  return (
    <UserContext.Provider value={{ profile, updateProfile, isSetUp: Boolean(profile.name) }}>
      {children}
    </UserContext.Provider>
  )
}

export function useUser() {
  const ctx = useContext(UserContext)
  if (!ctx) throw new Error('useUser must be used within UserProvider')
  return ctx
}
