import Icon from './Icon.jsx'

/**
 * Variants follow the design export.
 *
 * `primary` is #FC5100 with a near-black label, not white: white on this
 * orange is 3.32:1 and fails WCAG AA, while #09090b is 6.32:1. The export
 * makes the same choice. Do not "fix" the label to white.
 */
const variants = {
  primary: 'bg-cta text-on-cta hover:bg-primary-container',
  secondary:
    'bg-surface-raised text-text-primary border border-border hover:border-primary-container',
  ghost: 'text-text-muted hover:bg-surface-raised hover:text-primary',
  danger: 'bg-error text-on-error hover:opacity-90',
}

export default function Button({
  children,
  variant = 'primary',
  className = '',
  icon,
  ...rest
}) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-label-sm text-label-sm transition-colors duration-150 disabled:opacity-40 disabled:pointer-events-none ${variants[variant]} ${className}`}
      {...rest}
    >
      {icon && <Icon name={icon} size={18} />}
      {children}
    </button>
  )
}
