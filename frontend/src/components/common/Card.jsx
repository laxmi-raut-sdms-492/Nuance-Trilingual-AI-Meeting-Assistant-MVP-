export default function Card({ children, className = '', ...rest }) {
  return (
    <div
      className={`bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 shadow-card p-5 ${className}`}
      {...rest}
    >
      {children}
    </div>
  )
}
