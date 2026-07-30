import { useState, useEffect, useCallback } from 'react'
import toast from 'react-hot-toast'
import Icon from '../../components/common/Icon.jsx'
import Avatar from '../../components/common/Avatar.jsx'
import Loader from '../../components/common/Loader.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import AudioRecorder from '../../components/common/AudioRecorder.jsx'
import ConfirmDialog from '../../components/common/ConfirmDialog.jsx'
import { speakersApi, describeError } from '../../services/api.js'

/**
 * Ported from the design export (settings_speaker_enrollment).
 *
 * New UI over endpoints that already existed but had no screen. Once a voice
 * is enrolled, the diarizer labels matching segments with that person's name
 * instead of Speaker_00.
 *
 * These endpoints sit at the API root, not under /api, and have no auth in
 * front of them — anyone who can reach the backend can enroll a voice under
 * any name. Fine on localhost, not fine deployed.
 */

const SAMPLE_SECONDS = 6

export default function SpeakerEnrollment() {
  const [name, setName] = useState('')
  const [speakers, setSpeakers] = useState([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const { data } = await speakersApi.list()
      setSpeakers(data.speakers || [])
      setError(null)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleRecordingComplete = async (blob) => {
    const trimmed = name.trim()
    if (!trimmed) {
      toast.error('Enter a name before recording.')
      return
    }
    setSaving(true)
    try {
      await speakersApi.enroll(trimmed, blob)
      toast.success(`${trimmed} enrolled.`)
      setName('')
      await refresh()
    } catch (err) {
      toast.error(describeError(err))
    } finally {
      setSaving(false)
    }
  }

  const confirmDelete = async () => {
    setDeleting(true)
    try {
      await speakersApi.remove(pendingDelete)
      toast.success(`${pendingDelete} removed.`)
      setPendingDelete(null)
      await refresh()
    } catch (err) {
      toast.error(describeError(err))
    } finally {
      setDeleting(false)
    }
  }

  const nameReady = name.trim().length > 0

  return (
    <div className="flex flex-col gap-8">
      <section className="bg-surface border border-border rounded-xl p-6 flex flex-col gap-6 relative overflow-hidden">
        <div className="absolute -top-24 -right-24 w-48 h-48 bg-primary-container/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10">
          <label
            className="block font-label-sm text-label-sm text-text-muted mb-2 uppercase tracking-wider"
            htmlFor="speaker-name"
          >
            Speaker Name
          </label>
          <input
            id="speaker-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Priya Sharma"
            className="w-full bg-surface-container-low border border-border rounded-lg px-4 py-3 font-transcript-body text-transcript-body text-text-primary focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-faint"
          />
          <p className="font-meta-data text-meta-data text-text-muted mt-2">
            Record roughly {SAMPLE_SECONDS} seconds of this person speaking normally. Recording
            stops on its own.
          </p>
        </div>

        <div className="relative z-10 bg-surface-container-low border border-border rounded-lg">
          {saving ? (
            <Loader label="Enrolling voice..." />
          ) : nameReady ? (
            <AudioRecorder
              maxSeconds={SAMPLE_SECONDS}
              onRecordingComplete={handleRecordingComplete}
              hint="The sample is sent straight to the backend and stored as a voice embedding, not as audio."
            />
          ) : (
            <div className="h-32 flex items-center justify-center gap-1 p-4 opacity-50 relative overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center flex-col z-10">
                <Icon name="graphic_eq" className="text-text-faint mb-1" />
                <span className="font-meta-data text-meta-data text-text-faint">
                  Enter a name to start
                </span>
              </div>
              {[4, 8, 6, 12, 4, 8, 3, 10, 5, 7].map((h, i) => (
                <div
                  key={i}
                  className="waveform-bar"
                  style={{ height: `${h * 4}px`, animationDelay: `${(i + 1) / 10}s` }}
                />
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="flex flex-col gap-4">
        <h3 className="font-sidebar-header text-sidebar-header text-text-primary border-b border-border pb-2">
          Enrolled Voices
        </h3>

        {error && (
          <div className="rounded-lg border border-error/20 bg-error/10 px-4 py-3">
            <p className="font-label-sm text-label-sm text-error">Could not load enrolled voices</p>
            <p className="font-meta-data text-meta-data text-text-muted mt-1">{error}</p>
          </div>
        )}

        {loading ? (
          <Loader label="Loading voices..." />
        ) : speakers.length === 0 ? (
          <EmptyState
            icon="record_voice_over"
            title="No voices enrolled"
            subtitle="Without enrollment the diarizer still separates speakers, it just labels them Speaker_00, Speaker_01 and so on."
          />
        ) : (
          speakers.map((speaker) => (
            <div
              key={speaker}
              className="bg-surface border border-border rounded-xl p-4 flex items-center justify-between gap-4 hover:bg-surface-raised transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <Avatar name={speaker} size={40} />
                <div className="min-w-0">
                  <p className="text-text-primary truncate">{speaker}</p>
                  <p className="font-meta-data text-meta-data text-text-muted">
                    Voice profile enrolled
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setPendingDelete(speaker)}
                aria-label={`Remove ${speaker}`}
                className="text-text-muted hover:text-error transition-colors p-2 shrink-0"
              >
                <Icon name="delete" className="text-[18px]" />
              </button>
            </div>
          ))
        )}
      </section>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        busy={deleting}
        title="Remove Voice"
        message={`Remove ${pendingDelete}'s voice profile? Future meetings will label them Speaker_00 again. Existing transcripts are unchanged.`}
        confirmLabel="Remove"
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}
