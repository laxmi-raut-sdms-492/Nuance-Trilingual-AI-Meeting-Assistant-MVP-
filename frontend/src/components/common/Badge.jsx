/**
 * Status pills. The three meeting states map to the export's semantic
 * tokens — success / processing / error — so a status reads the same colour
 * everywhere it appears.
 */
const styles = {
  green: 'bg-success/10 text-success border border-success/20',
  yellow: 'bg-processing/10 text-processing border border-processing/20',
  red: 'bg-error/10 text-error border border-error/20',
  blue: 'bg-primary-container/10 text-primary border border-primary-container/20',
  gray: 'bg-surface-raised text-text-muted border border-border',
}

/** Meeting status -> badge colour, in one place so pages stay consistent. */
export const STATUS_COLORS = {
  Completed: 'green',
  Processing: 'yellow',
  Failed: 'red',
}

export default function Badge({ children, color = 'gray', className = '' }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-label-sm text-label-sm ${styles[color]} ${className}`}
    >
      {children}
    </span>
  )
}
