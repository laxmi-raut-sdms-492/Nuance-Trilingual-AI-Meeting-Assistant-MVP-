import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const ThemeContext = createContext(null)
const STORAGE_KEY = 'meetiq:theme'

function loadTheme() {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'dark'
  } catch {
    return false
  }
}

export function ThemeProvider({ children }) {
  const [dark, setDark] = useState(() => loadTheme())

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem(STORAGE_KEY, dark ? 'dark' : 'light')
    } catch {
      // ignore write failures
    }
  }, [dark])

  const toggleDark = useCallback(() => setDark((d) => !d), [])

  return (
    <ThemeContext.Provider value={{ dark, setDark, toggleDark }}>{children}</ThemeContext.Provider>
  )
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider')
  return ctx
}
