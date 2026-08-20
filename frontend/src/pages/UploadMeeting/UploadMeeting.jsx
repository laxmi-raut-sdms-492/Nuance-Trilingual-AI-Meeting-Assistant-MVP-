import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import Icon from '../../components/common/Icon.jsx'
import AudioRecorder from '../../components/common/AudioRecorder.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'
import { describeError } from '../../services/api.js'
import { SUPPORTED_LANGUAGE_NAMES } from '../../constants/languages.js'

/**
 * Ported from the design export (create_meeting_details),
 * create_meeting_source/ and create_meeting_upload_progress/ — one component,
 * three steps, as the routing table already had it.
 */

// Must stay a subset of the backend's ALLOWED_UPLOAD_EXTENSIONS. The browser
// recorder also produces .webm, which is accepted server-side but isn't
// offered here because no one picks it from a file dialog.
const ACCEPTED = ['.mp3', '.wav', '.mp4', '.m4a']

// The export's dropzone advertises "Max 500MB". The backend's MAX_UPLOAD_MB is
// 300, so this states 300 — promising a size the server rejects is worse than
// differing from the mockup.
const MAX_UPLOAD_MB = 300

const AGENDA_LIMIT = 500

function FieldError({ id, children }) {
  return (
    <div id={id} className="text-[11px] text-error font-medium flex items-center gap-1 mt-1">
      <Icon name="error" className="text-[14px]" />
      {children}
    </div>
  )
}

export default function UploadMeeting() {
  const navigate = useNavigate()
  const { addMeeting } = useMeetings()

  // step: 'details' -> 'source' (upload/record) -> in-progress/done
  const [step, setStep] = useState('details')

  const [title, setTitle] = useState('')
  const [agenda, setAgenda] = useState('')
  const [meetingType, setMeetingType] = useState('internal') // internal | client
  const [department, setDepartment] = useState('AI Team')
  const [projectName, setProjectName] = useState('')
  const [sttAdapter, setSttAdapter] = useState('local') // local | cloud
  const [formErrors, setFormErrors] = useState({})
  const [processingMode, setProcessingMode] = useState('local') // local | cloud

  const [mode, setMode] = useState('upload') // upload | record
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle') // idle | uploading | done | error
  const [savedMeeting, setSavedMeeting] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const inputRef = useRef(null)

  const details = { title, agenda, stt_adapter: sttAdapter, processingMode: sttAdapter, meetingType, department, projectName }

  const validateDetails = () => {
    const errors = {}
    if (!title.trim()) errors.title = 'Meeting Title is required.'
    if (!agenda.trim()) errors.agenda = 'Please provide an agenda for context.'
    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleContinue = (e) => {
    e.preventDefault()
    if (validateDetails()) setStep('source')
  }

  // Sends the real file to the backend, which stores it and queues
  // transcription. `progress` is the genuine number of bytes on the wire
  // reported by axios — an earlier build animated a fake timer that hit 100%
  // regardless of whether anything had actually been uploaded. Don't.
  const finishSave = async (f) => {
    setFile(f)
    setStatus('uploading')
    setProgress(0)
    setUploadError(null)

    try {
      const record = await addMeeting(f, details, (event) => {
        if (!event.total) return
        setProgress(Math.round((event.loaded / event.total) * 100))
      })
      setProgress(100)
      setSavedMeeting(record)
      setStatus('done')
      toast.success('Uploaded. Transcription is running now.')
      return record
    } catch (err) {
      const message = describeError(err)
      setUploadError(message)
      setStatus('error')
      toast.error(message)
      return null
    }
  }

  const startUpload = (f) => {
    const ext = '.' + f.name.split('.').pop().toLowerCase()
    if (!ACCEPTED.includes(ext)) {
      toast.error(`Unsupported file type. Accepted: ${ACCEPTED.join(', ')}`)
      return
    }
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      toast.error(`That file is over the ${MAX_UPLOAD_MB} MB limit.`)
      return
    }
    finishSave(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) startUpload(f)
  }

  const handleBrowse = (e) => {
    const f = e.target.files?.[0]
    if (f) startUpload(f)
  }

  const handleRecordingComplete = async (blob) => {
    const ext = blob.type.includes('mp4') ? 'm4a' : 'webm'
    const name = `Recording ${new Date().toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })}.${ext}`
    await finishSave(new File([blob], name, { type: blob.type }))
  }

  const reset = () => {
    setFile(null)
    setProgress(0)
    setStatus('idle')
    setSavedMeeting(null)
    setUploadError(null)
  }

  const startOver = () => {
    reset()
    setTitle('')
    setAgenda('')
    setMeetingType('internal')
    setDepartment('AI Team')
    setProjectName('')
    setFormErrors({})
    setProcessingMode('local')
    setStep('details')
  }

  const stepLabel =
    status !== 'idle'
      ? 'Step 3 of 3: Uploading & Processing'
      : step === 'details'
        ? 'Step 1 of 3: Details'
        : 'Step 2 of 3: Provide Source'

  return (
    <div className="w-full max-w-[768px] mx-auto flex flex-col gap-8">
      <div className="flex justify-between items-start gap-4">
        <div>
          <h1 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-text-primary">
            Create Meeting
          </h1>
          <p className="font-meta-data text-meta-data text-text-muted mt-1">{stepLabel}</p>
        </div>
        <button
          type="button"
          onClick={() => navigate('/meetings')}
          aria-label="Cancel and go back to meetings"
          className="text-text-muted hover:text-text-primary transition-colors p-2 rounded-lg hover:bg-surface-container"
        >
          <Icon name="close" className="text-2xl" />
        </button>
      </div>

      {/* ---------------------------------------------------------------- */}
      {step === 'details' && (
        <div className="bg-surface rounded-xl border border-border overflow-hidden relative">
          <div className="h-1 w-full bg-primary-container absolute top-0 left-0" />
          <form onSubmit={handleContinue} className="p-8 space-y-6" noValidate>
            <div className="space-y-2">
              <label
                className="block font-label-sm text-label-sm text-text-primary uppercase tracking-wider"
                htmlFor="meeting-title"
              >
                Meeting Title <span className="text-error">*</span>
              </label>
              <input
                id="meeting-title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Q3 Architecture & Sprint Sync"
                aria-invalid={Boolean(formErrors.title)}
                aria-describedby={formErrors.title ? 'title-error' : undefined}
                className={`w-full rounded-lg border input-base px-4 py-3 font-transcript-body text-transcript-body placeholder:text-text-faint transition-colors ${
                  formErrors.title ? 'input-error' : ''
                }`}
              />
              {formErrors.title && <FieldError id="title-error">{formErrors.title}</FieldError>}
            </div>

            {/* Meeting Type Selection: Internal vs Client */}
            <div className="space-y-2">
              <label className="block font-label-sm text-label-sm text-text-primary uppercase tracking-wider">
                Meeting Category
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setMeetingType('internal')}
                  className={`flex items-center justify-center gap-2 p-3 rounded-lg border text-sm font-medium transition-colors ${
                    meetingType === 'internal'
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border bg-surface text-text-muted hover:text-text-primary hover:border-text-muted'
                  }`}
                >
                  <Icon name="groups" className="text-lg" />
                  <span>👥 Internal Meeting</span>
                </button>
                <button
                  type="button"
                  onClick={() => setMeetingType('client')}
                  className={`flex items-center justify-center gap-2 p-3 rounded-lg border text-sm font-medium transition-colors ${
                    meetingType === 'client'
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border bg-surface text-text-muted hover:text-text-primary hover:border-text-muted'
                  }`}
                >
                  <Icon name="handshake" className="text-lg" />
                  <span>🤝 Client Meeting</span>
                </button>
              </div>
            </div>

            {/* Department & Project / Client Name Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label
                  className="block font-label-sm text-label-sm text-text-primary uppercase tracking-wider"
                  htmlFor="meeting-department"
                >
                  Department
                </label>
                <select
                  id="meeting-department"
                  value={department}
                  onChange={(e) => setDepartment(e.target.value)}
                  className="w-full rounded-lg border input-base px-4 py-3 font-transcript-body text-transcript-body bg-surface text-text-primary transition-colors cursor-pointer"
                >
                  <option value="AI Team">🤖 AI Team</option>
                  <option value="Software">💻 Software Engineering</option>
                  <option value="QA">🧪 QA & Testing</option>
                  <option value="Product & Design">🎨 Product & Design</option>
                  <option value="Management">📊 Management</option>
                  <option value="Sales & Marketing">🚀 Sales & Marketing</option>
                  <option value="Other">📁 Other</option>
                </select>
              </div>

              <div className="space-y-2">
                <label
                  className="block font-label-sm text-label-sm text-text-primary uppercase tracking-wider"
                  htmlFor="project-name"
                >
                  {meetingType === 'client' ? 'Client / Account Name' : 'Project Name'}
                </label>
                <input
                  id="project-name"
                  type="text"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder={meetingType === 'client' ? 'e.g. Acme Corp' : 'e.g. Nuance AI Assistant'}
                  className="w-full rounded-lg border input-base px-4 py-3 font-transcript-body text-transcript-body placeholder:text-text-faint transition-colors"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label
                className="font-label-sm text-label-sm text-text-primary uppercase tracking-wider flex justify-between"
                htmlFor="meeting-agenda"
              >
                <span>
                  Agenda <span className="text-error">*</span>
                </span>
                <span className="text-text-muted font-normal lowercase tracking-normal">
                  {agenda.length}/{AGENDA_LIMIT}
                </span>
              </label>
              <textarea
                id="meeting-agenda"
                value={agenda}
                maxLength={AGENDA_LIMIT}
                onChange={(e) => setAgenda(e.target.value)}
                placeholder="What will this meeting cover?"
                rows={3}
                aria-invalid={Boolean(formErrors.agenda)}
                aria-describedby={formErrors.agenda ? 'agenda-error' : undefined}
                className={`w-full rounded-lg border input-base px-4 py-3 font-transcript-body text-transcript-body placeholder:text-text-faint transition-colors resize-y ${
                  formErrors.agenda ? 'input-error' : ''
                }`}
              />
              {formErrors.agenda && <FieldError id="agenda-error">{formErrors.agenda}</FieldError>}
            </div>

            <div className="pt-6 border-t border-border flex justify-end items-center gap-3">
              <select
                id="stt-adapter-selector"
                value={sttAdapter}
                onChange={(e) => setSttAdapter(e.target.value)}
                className="bg-surface border border-border text-text-primary font-label-sm text-label-sm px-3 py-3 rounded-lg focus:outline-none focus:border-primary transition-colors cursor-pointer"
              >
                <option value="local">Local STT</option>
                <option value="cloud">Cloud STT</option>
              </select>
              <button
                type="submit"
                className="bg-cta text-on-cta font-label-sm text-label-sm px-6 py-3 rounded-lg hover:bg-primary-container transition-colors flex items-center gap-2"
              >
                Continue
                <Icon name="arrow_forward" size={18} />
              </button>
            </div>
          </form>
        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      {step === 'source' && (
        <>
          {status === 'idle' && (
            <>
              <button
                type="button"
                onClick={() => setStep('details')}
                className="group flex items-center gap-2 text-text-muted hover:text-text-primary transition-colors -mt-4"
              >
                <Icon name="arrow_back" className="text-lg" />
                <span className="font-meta-data text-meta-data">Back to details</span>
              </button>

              <div className="bg-primary/5 border border-primary/20 rounded-xl p-4 flex gap-3">
                <Icon name="translate" className="text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-label-sm text-label-sm text-text-primary">
                    Trilingual transcription
                  </p>
                  <p className="font-meta-data text-meta-data text-text-muted mt-1">
                    Speech is detected and transcribed automatically in {SUPPORTED_LANGUAGE_NAMES}.
                    Speakers can code-switch freely within one meeting.
                  </p>
                </div>
              </div>

              <div className="flex p-1 bg-surface-container rounded-lg w-fit border border-border">
                {[
                  { key: 'upload', icon: 'upload_file', label: 'Upload File' },
                  { key: 'record', icon: 'mic', label: 'Record Audio' },
                ].map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    onClick={() => setMode(m.key)}
                    className={`px-6 py-2 rounded font-label-sm text-label-sm flex items-center gap-2 transition-all ${
                      mode === m.key
                        ? 'bg-surface-raised text-text-primary shadow-sm'
                        : 'text-text-muted hover:text-text-primary hover:bg-surface/50'
                    }`}
                  >
                    <Icon name={m.icon} className="text-[16px]" />
                    {m.label}
                  </button>
                ))}
              </div>
            </>
          )}

          {status === 'idle' && mode === 'upload' && (
            <div
              role="button"
              tabIndex={0}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
              }}
              className={`relative group rounded-xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center py-20 px-6 text-center cursor-pointer ${
                dragOver
                  ? 'border-primary bg-primary/5'
                  : 'border-border bg-surface hover:border-primary hover:bg-primary/5'
              }`}
            >
              <div className="w-16 h-16 rounded-full bg-surface-raised group-hover:bg-primary/20 flex items-center justify-center mb-6 transition-colors border border-border group-hover:border-primary/50">
                <Icon
                  name="cloud_upload"
                  className="text-3xl text-text-muted group-hover:text-primary transition-colors"
                />
              </div>
              <h3 className="font-sidebar-header text-sidebar-header mb-2 text-text-primary">
                Drag &amp; drop your file here
              </h3>
              <p className="font-meta-data text-meta-data text-text-muted mb-6">
                or click to browse your computer
              </p>
              <div className="px-4 py-2 rounded-full bg-surface-container-high border border-border font-label-sm text-label-sm text-text-faint">
                Supported: MP3, WAV, M4A, MP4 (Max {MAX_UPLOAD_MB} MB)
              </div>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED.join(',')}
                className="hidden"
                onChange={handleBrowse}
              />
            </div>
          )}

          {status === 'idle' && mode === 'record' && (
            <div className="rounded-xl border border-border bg-surface p-8">
              <AudioRecorder onRecordingComplete={handleRecordingComplete} />
            </div>
          )}

          {status !== 'idle' && file && (
            <div className="bg-surface border border-border rounded-xl p-8 flex flex-col gap-6">
              <div className="flex items-center justify-between p-4 bg-surface-raised border border-border rounded-lg gap-3">
                <div className="flex items-center gap-4 min-w-0">
                  <div className="w-10 h-10 rounded-md bg-surface-container flex items-center justify-center text-primary shrink-0">
                    <Icon name="audio_file" />
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="font-meta-data text-meta-data text-text-primary truncate">
                      {file.name}
                    </span>
                    <span className="font-label-sm text-label-sm text-text-muted">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                </div>
                {status === 'done' ? (
                  <Icon name="check_circle" className="text-success shrink-0" />
                ) : (
                  <button
                    type="button"
                    onClick={reset}
                    aria-label="Cancel upload"
                    className="text-text-muted hover:text-error transition-colors p-1 shrink-0"
                  >
                    <Icon name="close" />
                  </button>
                )}
              </div>

              {status !== 'error' && (
                <div className="flex flex-col gap-3">
                  <div className="flex justify-between items-center font-meta-data text-meta-data">
                    <span className="text-text-primary">
                      {status === 'done' ? 'Upload complete' : 'Uploading...'}
                    </span>
                    <span className="text-primary font-semibold">{progress}%</span>
                  </div>
                  <div
                    className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden"
                    role="progressbar"
                    aria-valuenow={progress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <div
                      className="h-full bg-cta progress-bar-fill rounded-full"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="font-meta-data text-meta-data text-text-muted">
                    {status === 'done'
                      ? 'Transcription is running in the background — the meeting moves from Processing to Completed on its own.'
                      : 'Keep this tab open until the upload finishes.'}
                  </p>
                </div>
              )}

              {status === 'error' && (
                <div className="rounded-lg border border-error/20 bg-error/10 px-4 py-3">
                  <p className="font-label-sm text-label-sm text-error">Upload failed</p>
                  <p className="font-meta-data text-meta-data text-text-muted mt-1">{uploadError}</p>
                </div>
              )}

              <div className="flex flex-wrap gap-3 pt-2 border-t border-border">
                {status === 'error' && (
                  <button
                    type="button"
                    onClick={reset}
                    className="px-4 py-2.5 rounded-lg border border-border text-text-primary font-label-sm text-label-sm hover:bg-surface-raised transition-colors"
                  >
                    Try Again
                  </button>
                )}
                {status === 'done' && (
                  <>
                    <button
                      type="button"
                      onClick={startOver}
                      className="px-4 py-2.5 rounded-lg border border-border text-text-primary font-label-sm text-label-sm hover:bg-surface-raised transition-colors"
                    >
                      Create Another
                    </button>
                    {savedMeeting && (
                      <button
                        type="button"
                        onClick={() => navigate(`/meetings/${savedMeeting.id}`)}
                        className="px-6 py-2.5 rounded-lg bg-cta text-on-cta font-label-sm text-label-sm hover:bg-primary-container transition-colors flex items-center gap-2"
                      >
                        View Meeting
                        <Icon name="arrow_forward" size={18} />
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
