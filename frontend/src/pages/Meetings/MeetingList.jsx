import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import toast from 'react-hot-toast'
import Icon from '../../components/common/Icon.jsx'
import Loader from '../../components/common/Loader.jsx'
import Pagination from '../../components/common/Pagination.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import ConfirmDialog from '../../components/common/ConfirmDialog.jsx'
import { useMeetings, normalizeMeeting } from '../../context/MeetingsContext.jsx'
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
  const { meetings, removeMeeting, loading, error, refresh } = useMeetings()
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState(searchParams.get('q') || '')
  const [page, setPage] = useState(1)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [pendingPurge, setPendingPurge] = useState(null)
  const [pendingClearAll, setPendingClearAll] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [purging, setPurging] = useState(false)
  const [clearingAll, setClearingAll] = useState(false)
  const [restoringId, setRestoringId] = useState(null)

  const [trashMeetings, setTrashMeetings] = useState([])
  const [trashLoading, setTrashLoading] = useState(false)
  const [trashError, setTrashError] = useState(null)

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

  const isTrash = filter === 'trash'
  const term = query.trim()

  const loadTrash = useCallback(async () => {
    setTrashLoading(true)
    try {
      const { data } = await meetingsApi.trash()
      setTrashMeetings((data.meetings || []).map(normalizeMeeting))
      setTrashError(null)
    } catch (err) {
      setTrashError(describeError(err))
    } finally {
      setTrashLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isTrash) loadTrash()
  }, [isTrash, loadTrash])

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
    ? trashMeetings
    : results
      ? results.map((r) => byId.get(r.id) || { ...r, date: '—', time: '—', fileSizeLabel: '—' })
      : meetings

  const listLoading = isTrash ? trashLoading : loading
  const listError = isTrash ? trashError : error

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const confirmDelete = async () => {
    setDeleting(true)
    try {
      await removeMeeting(pendingDelete.id)
      toast.success('Meeting moved to Trash.')
      setPendingDelete(null)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setDeleting(false)
    }
  }

  const confirmPurge = async () => {
    setPurging(true)
    try {
      await meetingsApi.purge(pendingPurge.id)
      setTrashMeetings((prev) => prev.filter((m) => m.id !== pendingPurge.id))
      toast.success('Meeting permanently deleted.')
      setPendingPurge(null)
    } catch (err) {
      toast.error(describeError(err))
    } finally {
      setPurging(false)
    }
  }

  const confirmClearAll = async () => {
    setClearingAll(true)
    try {
      const { data } = await meetingsApi.purgeAll()
      setTrashMeetings([])
      setPage(1)
      toast.success(
        data.count === 1
          ? '1 meeting permanently deleted.'
          : `${data.count} meetings permanently deleted.`
      )
      setPendingClearAll(false)
    } catch (err) {
      toast.error(describeError(err))
    } finally {
      setClearingAll(false)
    }
  }

  const handleRestore = async (meeting) => {
    setRestoringId(meeting.id)
    try {
      await meetingsApi.restore(meeting.id)
      setTrashMeetings((prev) => prev.filter((m) => m.id !== meeting.id))
      await refresh({ quiet: true })
      toast.success('Meeting restored.')
    } catch (err) {
      toast.error(describeError(err))
    } finally {
      setRestoringId(null)
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

        {listError && (
          <div className="m-4 rounded-lg border border-error/20 bg-error/10 px-4 py-3">
            <p className="font-label-sm text-label-sm text-error">Could not load meetings</p>
            <p className="font-meta-data text-meta-data text-text-muted mt-1">{listError}</p>
          </div>
        )}

        {listLoading || (searching && !results) ? (
          <Loader label={searching ? 'Searching transcripts...' : isTrash ? 'Loading trash...' : 'Loading meetings...'} />
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
                ? 'Deleted meetings appear here. You can restore them or delete them permanently.'
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
                        {isTrash ? (
                          <div className="flex items-center gap-3">
                            <Icon name="graphic_eq" className="text-text-muted" />
                            <span className="text-text-primary font-medium">{m.title}</span>
                          </div>
                        ) : (
                          <Link to={`/meetings/${m.id}`} className="flex items-center gap-3">
                            <Icon name="graphic_eq" className="text-text-muted" />
                            <span className="text-text-primary font-medium hover:text-primary transition-colors">
                              {m.title}
                            </span>
                          </Link>
                        )}
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
                        <div className="flex justify-end gap-2">
                          {isTrash ? (
                            <>
                              <button
                                type="button"
                                onClick={() => handleRestore(m)}
                                disabled={restoringId === m.id}
                                aria-label={`Restore ${m.title}`}
                                className="text-text-muted hover:text-success transition-colors p-1 disabled:opacity-40"
                              >
                                <Icon name="restore" className="text-[18px]" />
                              </button>
                              <button
                                type="button"
                                onClick={() => setPendingPurge(m)}
                                aria-label={`Delete ${m.title} permanently`}
                                className="text-text-muted hover:text-error transition-colors p-1"
                              >
                                <Icon name="delete_forever" className="text-[18px]" />
                              </button>
                            </>
                          ) : (
                            <>
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
                            </>
                          )}
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
                    {isTrash ? (
                      <div className="flex items-center gap-3 min-w-0">
                        <Icon name="graphic_eq" className="text-text-muted shrink-0" />
                        <span className="text-text-primary font-medium truncate">{m.title}</span>
                      </div>
                    ) : (
                      <Link to={`/meetings/${m.id}`} className="flex items-center gap-3 min-w-0">
                        <Icon name="graphic_eq" className="text-text-muted shrink-0" />
                        <span className="text-text-primary font-medium truncate">{m.title}</span>
                      </Link>
                    )}
                    <StatusPill status={m.status} />
                  </div>
                  <MatchSnippets result={matchById?.get(m.id)} needle={term} />
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-meta-data text-meta-data text-text-faint">
                      {m.date} · {m.fileSizeLabel}
                    </p>
                    <div className="flex gap-1 shrink-0">
                      {isTrash ? (
                        <>
                          <button
                            type="button"
                            onClick={() => handleRestore(m)}
                            disabled={restoringId === m.id}
                            aria-label={`Restore ${m.title}`}
                            className="text-text-muted hover:text-success transition-colors p-2 disabled:opacity-40"
                          >
                            <Icon name="restore" className="text-[18px]" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setPendingPurge(m)}
                            aria-label={`Delete ${m.title} permanently`}
                            className="text-text-muted hover:text-error transition-colors p-2"
                          >
                            <Icon name="delete_forever" className="text-[18px]" />
                          </button>
                        </>
                      ) : (
                        <>
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
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination renders null for a single page unless a leading
                action (e.g. Clear All on Trash) is provided. */}
            {(totalPages > 1 || isTrash) && (
              <div className="p-4 border-t border-border bg-surface-raised/30">
                <Pagination
                  page={page}
                  totalPages={totalPages}
                  onChange={setPage}
                  leading={
                    isTrash ? (
                      <button
                        type="button"
                        onClick={() => setPendingClearAll(true)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-error/30 text-error font-label-sm text-label-sm hover:bg-error/10 transition-colors"
                      >
                        <Icon name="delete_forever" className="text-[16px]" />
                        Clear All
                      </button>
                    ) : null
                  }
                />
              </div>
            )}
          </>
        )}
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        busy={deleting}
        title="Move to Trash"
        message={`Move "${pendingDelete?.title}" to Trash? You can restore it later from the Trash page.`}
        confirmLabel="Move to Trash"
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />

      <ConfirmDialog
        open={Boolean(pendingPurge)}
        busy={purging}
        title="Delete Permanently"
        message={`Permanently delete "${pendingPurge?.title}"? The recording and transcript will be removed from the server. This cannot be undone.`}
        confirmLabel="Delete permanently"
        onConfirm={confirmPurge}
        onCancel={() => setPendingPurge(null)}
      />

      <ConfirmDialog
        open={pendingClearAll}
        busy={clearingAll}
        title="Clear All Trash"
        message={`Permanently delete all ${filtered.length} meetings in Trash? Their recordings and transcripts will be removed from the server. This cannot be undone.`}
        confirmLabel="Clear all"
        onConfirm={confirmClearAll}
        onCancel={() => setPendingClearAll(false)}
      />
    </>
  )
}
