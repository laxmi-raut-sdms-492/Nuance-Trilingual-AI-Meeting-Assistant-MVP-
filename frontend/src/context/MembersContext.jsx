import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { useUser } from './UserContext.jsx'

const MembersContext = createContext(null)
const STORAGE_KEY = 'meetiq:members'

function loadMembers() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function MembersProvider({ children }) {
  const { profile } = useUser()
  const [members, setMembers] = useState(() => loadMembers())

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(members))
    } catch {
      // ignore write failures
    }
  }, [members])

  // Keeps the admin's own card in sync with their profile (name/email edited
  // in Settings > Profile) automatically - no duplicated/hardcoded record.
  useEffect(() => {
    if (!profile.name) return
    setMembers((prev) => {
      const others = prev.filter((m) => m.id !== 'self')
      return [{ id: 'self', role: 'Admin', name: profile.name, email: profile.email }, ...others]
    })
  }, [profile.name, profile.email])

  const addMember = useCallback((member) => {
    const record = { id: `MEM-${Date.now()}`, role: 'Member', ...member }
    setMembers((prev) => [...prev, record])
    return record
  }, [])

  const removeMember = useCallback((id) => {
    setMembers((prev) => prev.filter((m) => m.id !== id))
  }, [])

  return (
    <MembersContext.Provider value={{ members, addMember, removeMember }}>
      {children}
    </MembersContext.Provider>
  )
}

export function useMembers() {
  const ctx = useContext(MembersContext)
  if (!ctx) throw new Error('useMembers must be used within MembersProvider')
  return ctx
}
