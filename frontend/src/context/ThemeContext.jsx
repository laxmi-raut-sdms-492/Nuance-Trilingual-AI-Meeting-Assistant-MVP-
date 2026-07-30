import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const ThemeContext = createContext(null)
const STORAGE_KEY = 'nuance:theme'

// Dark is the default, not light. The Stitch design is authored dark-only —
// light is a derived palette (see src/styles/tokens.css), so an unset
// preference should land on the theme the design was actually drawn for.
// index.html sets class="dark" up front to match, avoiding a white flash.
function loadTheme() {
  try {
    return localStorage.getItem(STORAGE_KEY) !== 'light'
  } catch {
    return true
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
