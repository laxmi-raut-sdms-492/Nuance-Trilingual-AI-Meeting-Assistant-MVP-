import { useState } from 'react'
import Button from '../../components/common/Button.jsx'
import Icon from '../../components/common/Icon.jsx'
import { useAuth } from '../../context/AuthContext.jsx'
import { describeError } from '../../services/api.js'

/** Shown instead of the app while the backend wants its password. */
export default function Login() {
  const { login } = useAuth()
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await login(password)
    } catch (err) {
      setError(err?.response?.status === 401 ? 'Wrong password.' : describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-background text-text-primary min-h-screen flex items-center justify-center p-gutter font-transcript-body">
      <form
        onSubmit={submit}
        className="w-full max-w-sm bg-surface border border-border rounded-xl p-6 flex flex-col gap-5"
      >
        <div className="flex items-center gap-3">
          <span className="w-10 h-10 rounded-lg bg-primary-container/15 flex items-center justify-center text-primary">
            <Icon name="lock" />
          </span>
          <div>
            <h1 className="font-label-sm text-label-sm uppercase tracking-wider text-text-muted">Nuance</h1>
            <p className="text-text-primary">Enter the app password to continue.</p>
          </div>
        </div>

        <label className="flex flex-col gap-2">
          <span className="font-label-sm text-label-sm text-text-muted uppercase tracking-wider">Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoFocus
            autoComplete="current-password"
            className="w-full bg-surface-container-low border border-border rounded-lg px-4 py-3 text-text-primary focus:outline-none focus:border-primary-container transition-colors"
          />
        </label>

        {error && <p className="text-error font-meta-data text-meta-data">{error}</p>}

        <Button type="submit" disabled={busy || !password} icon="login">
          {busy ? 'Signing in…' : 'Sign in'}
        </Button>
      </form>
    </div>
  )
}
