import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <p className="text-6xl font-extrabold text-primary-600">404</p>
      <p className="text-lg font-semibold text-gray-800 mt-2">Page not found</p>
      <p className="text-sm text-gray-400 mt-1">The page you're looking for doesn't exist.</p>
      <Link to="/" className="mt-6 px-4 py-2 rounded-xl bg-primary-600 text-white text-sm font-semibold">
        Back to All Meetings
      </Link>
    </div>
  )
}
