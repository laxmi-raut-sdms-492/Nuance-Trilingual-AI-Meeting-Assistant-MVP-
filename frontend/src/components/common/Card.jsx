/**
 * The export's standard container: bg-surface, 1px border, rounded-xl.
 * Padding varies by context (p-5 on stat cards, p-6 on panels), so it is a
 * prop rather than baked in.
 */
export default function Card({ children, className = '', padding = 'p-6', ...rest }) {
  return (
    <div
      className={`bg-surface border border-border rounded-xl ${padding} ${className}`}
      {...rest}
    >
      {children}
    </div>
  )
}
