import { ChevronLeft, ChevronRight } from 'lucide-react'

export default function Pagination({ page, totalPages, onChange }) {
  return (
    <div className="flex items-center justify-between pt-4">
      <p className="text-xs text-gray-400">
        Page {page} of {totalPages}
      </p>
      <div className="flex items-center gap-2">
        <button
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
          className="w-8 h-8 flex items-center justify-center rounded-lg border border-gray-200 text-gray-500 disabled:opacity-40 hover:bg-gray-50"
        >
          <ChevronLeft size={14} />
        </button>
        {Array.from({ length: totalPages }).map((_, i) => (
          <button
            key={i}
            onClick={() => onChange(i + 1)}
            className={`w-8 h-8 rounded-lg text-xs font-semibold ${
              page === i + 1 ? 'bg-primary-600 text-white' : 'text-gray-500 hover:bg-gray-50 border border-gray-200'
            }`}
          >
            {i + 1}
          </button>
        ))}
        <button
          disabled={page >= totalPages}
          onClick={() => onChange(page + 1)}
          className="w-8 h-8 flex items-center justify-center rounded-lg border border-gray-200 text-gray-500 disabled:opacity-40 hover:bg-gray-50"
        >
          <ChevronRight size={14} />
        </button>
      </div>
    </div>
  )
}
