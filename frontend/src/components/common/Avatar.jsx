const PALETTE = [
  '#6366f1', '#22c55e', '#f59e0b', '#ec4899',
  '#14b8a6', '#8b5cf6', '#ef4444', '#0ea5e9'
]

function colorFor(seed) {
  const s = seed || '?'
  let hash = 0
  for (let i = 0; i < s.length; i++) hash = s.charCodeAt(i) + ((hash << 5) - hash)
  return PALETTE[Math.abs(hash) % PALETTE.length]
}

function initialsFor(name) {
  if (!name) return '?'
  const parts = name.trim().split(/\s+/)
  const first = parts[0]?.[0] || ''
  const last = parts.length > 1 ? parts[parts.length - 1][0] : ''
  return (first + last).toUpperCase()
}

export default function Avatar({ name, size = 36, className = '' }) {
  return (
    <div
      className={`rounded-full flex items-center justify-center shrink-0 font-semibold text-white ${className}`}
      style={{
        width: size,
        height: size,
        backgroundColor: colorFor(name),
        fontSize: size * 0.4
      }}
    >
      {initialsFor(name)}
    </div>
  )
}
