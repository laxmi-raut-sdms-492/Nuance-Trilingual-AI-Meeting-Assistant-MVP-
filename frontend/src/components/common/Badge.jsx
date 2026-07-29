const styles = {
  green: 'bg-green-50 text-green-600',
  yellow: 'bg-amber-50 text-amber-600',
  red: 'bg-red-50 text-red-600',
  blue: 'bg-primary-50 text-primary-600',
  gray: 'bg-gray-100 text-gray-600'
}

export default function Badge({ children, color = 'gray' }) {
  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${styles[color]}`}>
      {children}
    </span>
  )
}
