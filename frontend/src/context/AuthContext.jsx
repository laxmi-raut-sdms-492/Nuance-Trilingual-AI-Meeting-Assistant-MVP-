import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { authApi, getToken, setToken } from '../services/api.js'

/**
 * The one shared password the backend may ask for (backend/auth.py).
 *
 * On load it asks the server whether a password is set at all; a solo dev
 * setup with no APP_PASSWORD never sees the sign-in screen. The token is
 * kept in localStorage so a reload does not ask again, and a 401 from any
 * request (password changed on the server) drops it and shows the screen.
 */
const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [required, setRequired] = useState(null) // null = not asked yet
  const [token, setTokenState] = useState(() => getToken())

  useEffect(() => {
    let cancelled = false
    authApi
      .status()
      .then(({ data }) => {
        if (!cancelled) setRequired(Boolean(data?.required))
      })
      .catch(() => {
        // Backend unreachable: let the app render; its own errors will say so.
        if (!cancelled) setRequired(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const onUnauthorized = () => {
      setToken(null)
      setTokenState(null)
      setRequired(true)
    }
    window.addEventListener('nuance:unauthorized', onUnauthorized)
    return () => window.removeEventListener('nuance:unauthorized', onUnauthorized)
  }, [])

  const login = useCallback(async (password) => {
    const { data } = await authApi.login(password)
    setToken(data.token)
    setTokenState(data.token)
    setRequired(Boolean(data.required))
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setTokenState(null)
  }, [])

  const signedIn = required === false || Boolean(token)

  return (
    <AuthContext.Provider value={{ required, signedIn, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
