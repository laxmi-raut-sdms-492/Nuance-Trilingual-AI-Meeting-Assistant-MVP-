import { useEffect, useRef } from 'react'
import Icon from './Icon.jsx'

/**
 * Ported from the delete modal in the design export (all_meetings).
 * Replaces the old `window.confirm`, which the brief called out.
 *
 * Escape closes, the backdrop closes, and focus moves to the confirm button on
 * open so the dialog is usable from the keyboard — none of which the exported
 * HTML does, since it toggles a class from an inline onclick.
 */
export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Delete',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  busy = false,
}) {
  const confirmRef = useRef(null)

  useEffect(() => {
    if (!open) return
    confirmRef.current?.focus()
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel?.()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onCancel])

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/80 backdrop-blur-sm p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel?.()
      }}
    >
      <div className="bg-surface border border-border rounded-xl p-6 max-w-sm w-full shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-error/20 via-error to-error/20" />
        <div className="flex items-center gap-3 mb-4 text-error">
          <Icon name="warning" className="text-[28px]" />
          <h3
            id="confirm-dialog-title"
            className="font-sidebar-header text-sidebar-header font-bold text-text-primary"
          >
            {title}
          </h3>
        </div>
        <p className="font-transcript-body text-transcript-body text-text-muted mb-6">{message}</p>
        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="px-4 py-2 rounded-lg border border-border text-text-primary font-label-sm text-label-sm hover:bg-surface-raised transition-colors disabled:opacity-40"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            ref={confirmRef}
            onClick={onConfirm}
            disabled={busy}
            className="px-4 py-2 rounded-lg bg-error text-on-error font-label-sm text-label-sm font-bold hover:bg-error/90 transition-colors disabled:opacity-40"
          >
            {busy ? 'Deleting…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
