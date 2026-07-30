import { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import toast from 'react-hot-toast'
import Icon from '../../components/common/Icon.jsx'
import Loader from '../../components/common/Loader.jsx'
import Pagination from '../../components/common/Pagination.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import ConfirmDialog from '../../components/common/ConfirmDialog.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'
import { meetingsApi, describeError } from '../../services/api.js'

/**
 * Ported from the design export (all_meetings).
 *
 * The table becomes stacked cards below md: — a five-column table is unusable
 * on a phone, and the export's mobile screens use cards throughout.
 *
 * Search runs on the server (GET /api/search), not over the loaded list. The
 * list response omits transcript bodies, so a client-side filter can only ever
 * match titles; the backend searches transcript text as well and returns the
 * matching lines, which is the point of having transcripts at all.
 */

// Typing a word shouldn't fire a query per keystroke.
const SEARCH_DEBOUNCE_MS = 250

const STATUS_PILL = {
  Completed: 'bg-success/10 text-success border-success/20',
  Processing: 'bg-processing/10 text-processing border-processing/20',
  Failed: 'bg-error/10 text-error border-error/20',
}

const PAGE_SIZE = 8

/**
 * Highlight the query inside a matched line.
 *
 * Split-and-rebuild rather than innerHTML: transcript text is user-supplied
 * and must never be interpreted as markup. The needle is escaped before it
 * reaches the RegExp, so a query like "c++" is matched literally.
 */
function Highlight({ text, needle }) {
  const term = needle.trim()
  if (!term) return text
  const parts = String(text).split(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === term.toLowerCase() ? (
      <mark key={i} className="bg-primary-container/30 text-text-primary rounded px-0.5">
        {part}
      </mark>
    ) : (
      part
    )
  )
}

/** The transcript lines that matched, shown under the meeting they belong to. */
function MatchSnippets({ result, needle }) {
  if (!result?.matches?.length) return null
  const hidden = result.matchCount - result.matches.length
  return (
    <div className="mt-2 space-y-1 border-l-2 border-border pl-3">
      {result.matches.map((m, i) => (
        <p key={i} className="font-meta-data text-meta-data text-text-faint">
          <span className="tabular-nums">{m.time}</span>
          <span className="mx-1">·</span>
          <span style={{ color: m.color || undefined }}>{m.speaker}</span>
          <span className="mx-1">·</span>
          <span className="text-text-muted">
            <Highlight text={m.text} needle={needle} />
          </span>
        </p>
      ))}
      {hidden > 0 && (
        <p className="font-meta-data text-meta-data text-text-faint">
          +{hidden} more {hidden === 1 ? 'line' : 'lines'} in this meeting
        </p>
      )}
    </div>
  )
}

function StatusPill({ status }) {
  return (
    <span
      className={`px-2 py-1 rounded text-[10px] font-bold tracking-wide border ${
        STATUS_PILL[status] || 'bg-surface-raised text-text-muted border-border'
      }`}
    >
      {status?.toUpperCase()}
    </span>
  )
}

export default function MeetingList({ filter }) {
  const navigate = useNavigate()
  const { meetings, removeMeeting, loading, error } = useMeetings()
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState(searchParams.get('q') || '')
  const [page, setPage] = useState(1)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)

  // null while not searching; an array (possibly empty) once the server has
  // answered. The distinction matters — an empty array is "no matches", null
  // is "show everything".
  const [results, setResults] = useState(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState(null)
  const searchSeq = useRef(0)

  // Keep the search box in sync if the user searches again from the Navbar.
  useEffect(() => {
    const q = searchParams.get('q') || ''
    setQuery(q)
    setPage(1)
  }, [searchParams])

  // Trash has no soft-delete backing yet. Shown honestly as empty rather than
  // faking rows — see README "Known limitations".
  const isTrash = filter === 'trash'
  const term = query.trim()

  useEffect(() => {
    if (isTrash || !term) {
      searchSeq.current++ // discard anything still on the wire
      setResults(null)
      setSearching(false)
      setSearchError(null)
      return
    }

    const seq = ++searchSeq.current
    setSearching(true)
    const timer = setTimeout(async () => {
      try {
        const { data } = await meetingsApi.search(term)
        if (seq !== searchSeq.current) return
        setResults(data.results || [])
        setSearchError(null)
      } catch (err) {
        if (seq !== searchSeq.current) return
        setResults([])
        setSearchError(describeError(err))
      } finally {
        if (seq === searchSeq.current) setSearching(false)
      }
    }, SEARCH_DEBOUNCE_MS)

    return () => clearTimeout(timer)
  }, [term, isTrash])

  // Search returns id/title/status/matches. Everything else the row renders
  // (formatted date, file size) comes from the already-loaded list; a result
  // for a meeting the list has not caught up with still renders, minus those.
  const byId = new Map(meetings.map((m) => [m.id, m]))
  const matchById = results ? new Map(results.map((r) => [r.id, r])) : null

  const filtered = isTrash
    ? []
    : results
      ? results.map((r) => byId.get(r.id) || { ...r, date: '—', time: '—', fileSizeLabel: '—' })
      : meetings

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const confirmDelete = async () => {
    setDeleting(true)
    try {
      await removeMeeting(pendingDelete.id)
      toast.success('Meeting deleted.')
      setPendingDelete(null)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div>
          <p className="font-meta-data text-meta-data text-text-muted mb-1">
            {results
              ? `${filtered.length} ${filtered.length === 1 ? 'match' : 'matches'} for "${term}"`
              : `${filtered.length} ${filtered.length === 1 ? 'meeting' : 'meetings'}`}
          </p>
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
            {isTrash ? 'Trash' : 'All Meetings'}
          </h2>
        </div>
        {!isTrash && (
          <button
            type="button"
            onClick={() => navigate('/upload')}
            className="bg-cta hover:bg-primary-container text-on-cta font-label-sm text-label-sm py-3 px-6 rounded-lg flex items-center gap-2 transition-all hover:scale-105"
          >
            <Icon name="add" />
            Create Meeting
          </button>
        )}
      </div>

      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        {!isTrash && (
          <div className="p-4 border-b border-border">
            <div className="relative max-w-sm">
              <Icon
                name="search"
                size={18}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
              />
              <input
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value)
                  setPage(1)
                  setSearchParams(e.target.value ? { q: e.target.value } : {})
                }}
                placeholder="Search titles, agendas and transcripts..."
                className="input-base w-full pl-10 pr-10 py-2 rounded-lg border font-meta-data text-meta-data placeholder:text-text-faint"
              />
              {searching && (
                <Icon
                  name="progress_activity"
                  size={18}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted animate-spin pointer-events-none"
                />
              )}
            </div>
          </div>
        )}

        {searchError && (
          <div className="m-4 rounded-lg border border-error/20 bg-error/10 px-4 py-3">
            <p className="font-label-sm text-label-sm text-error">Search failed</p>
            <p className="font-meta-data text-meta-data text-text-muted mt-1">{searchError}</p>
          </div>
        )}

        {error && (
          <div className="m-4 rounded-lg border border-error/20 bg-error/10 px-4 py-3">
            <p className="font-label-sm text-label-sm text-error">Could not load meetings</p>
            <p className="font-meta-data text-meta-data text-text-muted mt-1">{error}</p>
          </div>
        )}

        {loading || (searching && !results) ? (
          <Loader label={searching ? 'Searching transcripts...' : 'Loading meetings...'} />
        ) : filtered.length === 0 ? (
          <EmptyState
            icon={isTrash ? 'delete' : results ? 'search_off' : 'event_note'}
            title={
              isTrash
                ? 'Trash is empty'
                : results
                  ? `No meetings match "${term}"`
                  : 'No meetings uploaded yet'
            }
            subtitle={
              isTrash
                ? 'Deleting a meeting removes it and its recording immediately — there is no soft delete yet, so nothing collects here.'
                : results
                  ? 'Titles, agendas and every transcribed line were searched, in all three languages.'
                  : 'Upload an audio or video recording to get started.'
            }
          />
        ) : (
          <>
            {/* Desktop table */}
            <div className="hidden md:block overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-border bg-surface-raised/50">
                    {['Title', 'Date', 'File Size', 'Status'].map((h) => (
                      <th
                        key={h}
                        className="p-4 font-label-sm text-label-sm text-text-muted uppercase tracking-wider"
                      >
                        {h}
                      </th>
                    ))}
                    <th className="p-4 font-label-sm text-label-sm text-text-muted uppercase tracking-wider text-right">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="font-meta-data text-meta-data divide-y divide-border">
                  {paged.map((m) => (
                    <tr key={m.id} className="hover:bg-surface-raised transition-colors group">
                      <td className="p-4 align-top">
                        <Link to={`/meetings/${m.id}`} className="flex items-center gap-3">
                          <Icon name="graphic_eq" className="text-text-muted" />
                          <span className="text-text-primary font-medium hover:text-primary transition-colors">
                            {m.title}
                          </span>
                        </Link>
                        <MatchSnippets result={matchById?.get(m.id)} needle={term} />
                      </td>
                      <td className="p-4 align-top text-text-faint whitespace-nowrap">
                        {m.date} · {m.time}
                      </td>
                      <td className="p-4 align-top text-text-faint whitespace-nowrap">{m.fileSizeLabel}</td>
                      <td className="p-4 align-top">
                        <StatusPill status={m.status} />
                      </td>
                      <td className="p-4 align-top text-right">
                        <div className="flex justify-end gap-2 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                          <button
                            type="button"
                            onClick={() => navigate(`/meetings/${m.id}`)}
                            aria-label={`View ${m.title}`}
                            className="text-text-muted hover:text-primary transition-colors p-1"
                          >
                            <Icon name="visibility" className="text-[18px]" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setPendingDelete(m)}
                            aria-label={`Delete ${m.title}`}
                            className="text-text-muted hover:text-error transition-colors p-1"
                          >
                            <Icon name="delete" className="text-[18px]" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile cards */}
            <div className="md:hidden divide-y divide-border">
              {paged.map((m) => (
                <div key={m.id} className="p-4 flex flex-col gap-3">
                  <div className="flex items-start justify-between gap-3">
                    <Link to={`/meetings/${m.id}`} className="flex items-center gap-3 min-w-0">
                      <Icon name="graphic_eq" className="text-text-muted shrink-0" />
                      <span className="text-text-primary font-medium truncate">{m.title}</span>
                    </Link>
                    <StatusPill status={m.status} />
                  </div>
                  <MatchSnippets result={matchById?.get(m.id)} needle={term} />
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-meta-data text-meta-data text-text-faint">
                      {m.date} · {m.fileSizeLabel}
                    </p>
                    <div className="flex gap-1 shrink-0">
                      <button
                        type="button"
                        onClick={() => navigate(`/meetings/${m.id}`)}
                        aria-label={`View ${m.title}`}
                        className="text-text-muted hover:text-primary transition-colors p-2"
                      >
                        <Icon name="visibility" className="text-[18px]" />
                      </button>
                      <button
                        type="button"
                        onClick={() => setPendingDelete(m)}
                        aria-label={`Delete ${m.title}`}
                        className="text-text-muted hover:text-error transition-colors p-2"
                      >
                        <Icon name="delete" className="text-[18px]" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination renders null for a single page — don't leave an
                empty bordered strip behind when it does. */}
            {totalPages > 1 && (
              <div className="p-4 border-t border-border bg-surface-raised/30">
                <Pagination page={page} totalPages={totalPages} onChange={setPage} />
              </div>
            )}
          </>
        )}
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        busy={deleting}
        title="Delete Meeting"
        message={`Delete "${pendingDelete?.title}"? This cannot be undone — the recording and its transcript are removed from the server.`}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </>
  )
}
