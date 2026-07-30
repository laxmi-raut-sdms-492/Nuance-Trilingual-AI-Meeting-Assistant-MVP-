import Icon from './Icon.jsx'

export default function Pagination({ page, totalPages, onChange }) {
  if (totalPages <= 1) return null

  const arrow =
    'w-8 h-8 flex items-center justify-center rounded-lg border border-border text-text-muted disabled:opacity-40 hover:bg-surface-raised hover:text-primary transition-colors'

  return (
    <div className="flex items-center justify-between pt-4">
      <p className="font-meta-data text-meta-data text-text-muted">
        Page {page} of {totalPages}
      </p>
      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
          aria-label="Previous page"
          className={arrow}
        >
          <Icon name="chevron_left" size={16} />
        </button>
        {Array.from({ length: totalPages }).map((_, i) => (
          <button
            key={i}
            type="button"
            onClick={() => onChange(i + 1)}
            aria-current={page === i + 1 ? 'page' : undefined}
            className={`w-8 h-8 rounded-lg font-label-sm text-label-sm transition-colors ${
              page === i + 1
                ? 'bg-cta text-on-cta'
                : 'text-text-muted border border-border hover:bg-surface-raised'
            }`}
          >
            {i + 1}
          </button>
        ))}
        <button
          type="button"
          disabled={page >= totalPages}
          onClick={() => onChange(page + 1)}
          aria-label="Next page"
          className={arrow}
        >
          <Icon name="chevron_right" size={16} />
        </button>
      </div>
    </div>
  )
}
