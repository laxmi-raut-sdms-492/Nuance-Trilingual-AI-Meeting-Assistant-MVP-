const variants = {
  primary: 'bg-primary-600 text-white hover:bg-primary-700',
  secondary: 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-50',
  ghost: 'text-gray-600 hover:bg-gray-100'
}

export default function Button({ children, variant = 'primary', className = '', icon: Icon, ...rest }) {
  return (
    <button
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-colors duration-200 ${variants[variant]} ${className}`}
      {...rest}
    >
      {Icon && <Icon size={16} />}
      {children}
    </button>
  )
}
