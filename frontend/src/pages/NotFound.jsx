import { Link } from 'react-router-dom'
import Icon from '../components/common/Icon.jsx'

/** Ported from the design export (404_not_found). */
export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <p className="text-[120px] md:text-[160px] leading-none font-headline-lg font-bold text-accent tracking-tighter mb-4">
        404
      </p>
      <p className="font-headline-lg text-headline-lg text-text-primary">Page not found</p>
      <p className="font-meta-data text-meta-data text-text-muted mt-2 max-w-md">
        The intelligence module you are looking for has been moved, deleted, or possibly never
        existed.
      </p>
      <Link
        to="/dashboard"
        className="mt-8 inline-flex items-center gap-2 bg-cta text-on-cta font-label-sm text-label-sm py-3 px-6 rounded-lg hover:bg-primary-container transition-colors"
      >
        <Icon name="dashboard" size={18} />
        Back to Dashboard
      </Link>
    </div>
  )
}
