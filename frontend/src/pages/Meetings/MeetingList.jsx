import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Search, Upload, Trash2 } from 'lucide-react'
import Card from '../../components/common/Card.jsx'
import Badge from '../../components/common/Badge.jsx'
import Button from '../../components/common/Button.jsx'
import Pagination from '../../components/common/Pagination.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'

const statusColor = { Completed: 'green', Processing: 'yellow', Failed: 'red' }
const PAGE_SIZE = 8

const titles = {
  trash: 'Trash'
}

export default function MeetingList({ filter }) {
  const navigate = useNavigate()
  const { meetings, removeMeeting } = useMeetings()
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState(searchParams.get('q') || '')
  const [page, setPage] = useState(1)

  // Keep the search box in sync if the user searches again from the Navbar.
  useEffect(() => {
    const q = searchParams.get('q') || ''
    setQuery(q)
    setPage(1)
  }, [searchParams])

  // "Trash" has no real soft-delete data source yet - shown honestly as
  // empty until that feature is wired up, rather than faking entries.
  const source = filter === 'trash' ? [] : meetings

  const filtered = source.filter((m) => m.title.toLowerCase().includes(query.toLowerCase()))
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{titles[filter] || 'All Meetings'}</h1>
          <p className="text-sm text-gray-400 mt-1">{filtered.length} meetings found</p>
        </div>
        <Button icon={Upload} onClick={() => navigate('/upload')}>
          Create Meeting
        </Button>
      </div>

      <Card>
        <div className="relative max-w-sm mb-4">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setPage(1)
              setSearchParams(e.target.value ? { q: e.target.value } : {})
            }}
            placeholder="Search meetings..."
            className="w-full pl-9 pr-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-200"
          />
        </div>

        {filtered.length === 0 ? (
          <EmptyState
            title={filter === 'trash' ? 'Nothing here yet' : 'No meetings uploaded yet'}
            subtitle={
              filter === 'trash'
                ? 'Deleted meetings will appear here.'
                : 'Upload an audio or video recording to get started.'
            }
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-100 dark:border-gray-700">
                    <th className="pb-3 font-medium">Meeting Title</th>
                    <th className="pb-3 font-medium">Date</th>
                    <th className="pb-3 font-medium">File</th>
                    <th className="pb-3 font-medium">Status</th>
                    <th className="pb-3 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {paged.map((m) => (
                    <tr key={m.id} className="border-b border-gray-50 dark:border-gray-800 last:border-0 hover:bg-gray-50/60 dark:hover:bg-gray-800/60">
                      <td className="py-3 font-semibold text-gray-800 dark:text-gray-100">{m.title}</td>
                      <td className="py-3 text-gray-500 dark:text-gray-400">
                        {m.date} <span className="text-gray-300 dark:text-gray-600">·</span> {m.time}
                      </td>
                      <td className="py-3 text-gray-500 dark:text-gray-400">{m.fileSizeLabel}</td>
                      <td className="py-3">
                        <Badge color={statusColor[m.status] || 'gray'}>{m.status}</Badge>
                      </td>
                      <td className="py-3 text-right">
                        <div className="flex items-center justify-end gap-3">
                          <button
                            onClick={() => navigate(`/meetings/${m.id}`)}
                            className="text-primary-600 font-semibold hover:underline"
                          >
                            View Details
                          </button>
                          <button
                            onClick={() => removeMeeting(m.id)}
                            className="text-gray-300 hover:text-red-500"
                            title="Delete"
                          >
                            <Trash2 size={15} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} totalPages={totalPages} onChange={setPage} />
          </>
        )}
      </Card>
    </div>
  )
}
