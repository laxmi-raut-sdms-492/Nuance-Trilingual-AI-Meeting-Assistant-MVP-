import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { UploadCloud, FileAudio, X, CheckCircle2, Mic, ArrowLeft } from 'lucide-react'
import toast from 'react-hot-toast'
import Card from '../../components/common/Card.jsx'
import Button from '../../components/common/Button.jsx'
import AudioRecorder from '../../components/common/AudioRecorder.jsx'
import { useMeetings } from '../../context/MeetingsContext.jsx'
import { saveAudioBlob } from '../../utils/audioStore.js'

const ACCEPTED = ['.mp3', '.wav', '.mp4', '.m4a']

export default function UploadMeeting() {
  const navigate = useNavigate()
  const { addMeeting } = useMeetings()

  // step: 'details' -> 'source' (upload/record) -> in-progress/done
  const [step, setStep] = useState('details')

  // Meeting details form state
  const [title, setTitle] = useState('')
  const [agenda, setAgenda] = useState('')
  const [formErrors, setFormErrors] = useState({})

  const [mode, setMode] = useState('upload') // upload | record
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle') // idle | uploading | done
  const [savedMeeting, setSavedMeeting] = useState(null)
  const inputRef = useRef(null)

  const details = { title, agenda }

  const validateDetails = () => {
    const errors = {}
    if (!title.trim()) errors.title = 'Meeting title is required'
    if (!agenda.trim()) errors.agenda = 'Meeting agenda is required'
    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleContinue = (e) => {
    e.preventDefault()
    if (validateDetails()) setStep('source')
  }

  // Saves the real file/blob as a meeting record AND persists the actual
  // audio bytes to IndexedDB so it can be played back later from the
  // meeting details page. This only ever runs once per upload/recording -
  // it's called directly from a plain callback (not from inside a setState
  // updater), which avoids React 18 Strict Mode's dev-time double-invoke of
  // updater functions - that double-invoke was the cause of "1 file shows
  // as 2" in the dashboard counts.
  const finishSave = async (f, blob) => {
    const record = addMeeting(f, details)
    await saveAudioBlob(record.id, blob || f)
    setSavedMeeting(record)
    setStatus('done')
    return record
  }

  const startUpload = (f) => {
    const ext = '.' + f.name.split('.').pop().toLowerCase()
    if (!ACCEPTED.includes(ext)) {
      toast.error(`Unsupported file type. Accepted: ${ACCEPTED.join(', ')}`)
      return
    }

    setFile(f)
    setStatus('uploading')
    setProgress(0)

    let current = 0
    const interval = setInterval(() => {
      current += 8
      if (current >= 100) {
        current = 100
        clearInterval(interval)
        setProgress(100)
        finishSave(f).then(() => {
          toast.success('File uploaded. It now appears in All Meetings.')
        })
      } else {
        setProgress(current)
      }
    }, 150)
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

  const handleRecordingComplete = async (blob, seconds) => {
    const ext = blob.type.includes('mp4') ? 'm4a' : 'webm'
    const name = `Recording ${new Date().toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })}.${ext}`
    const recordedFile = new File([blob], name, { type: blob.type })
    setFile(recordedFile)
    setStatus('uploading')
    setProgress(100)
    await finishSave(recordedFile, blob)
    toast.success('Recording saved. It now appears in All Meetings.')
  }

  const reset = () => {
    setFile(null)
    setProgress(0)
    setStatus('idle')
    setSavedMeeting(null)
  }

  const startOver = () => {
    reset()
    setTitle('')
    setAgenda('')
    setFormErrors({})
    setStep('details')
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Create Meeting</h1>
        <p className="text-sm text-gray-400 mt-1">
          {step === 'details'
            ? 'Start by giving your meeting a title, type, and agenda.'
            : 'Upload a recording or record one live. Files are saved to your meeting list right in this browser — connect the backend from the FSD to generate real transcripts and summaries.'}
        </p>
      </div>

      {step === 'details' && (
        <Card>
          <form onSubmit={handleContinue} className="space-y-4">
            <div>
              <label htmlFor="meeting-title" className="block text-sm font-semibold text-gray-800 mb-1.5">
                Meeting Title <span className="text-red-500">*</span>
              </label>
              <input
                id="meeting-title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Q3 Marketing Sync"
                className={`w-full rounded-xl border px-4 py-2.5 text-sm text-gray-800 placeholder:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-400/40 ${
                  formErrors.title ? 'border-red-300' : 'border-gray-200 focus:border-primary-400'
                }`}
              />
              {formErrors.title && <p className="text-xs text-red-500 mt-1">{formErrors.title}</p>}
            </div>

            <div>
              <label htmlFor="meeting-agenda" className="block text-sm font-semibold text-gray-800 mb-1.5">
                Agenda <span className="text-red-500">*</span>
              </label>
              <textarea
                id="meeting-agenda"
                value={agenda}
                onChange={(e) => setAgenda(e.target.value)}
                placeholder="What will this meeting cover?"
                rows={2}
                className={`w-full rounded-xl border px-4 py-2.5 text-sm text-gray-800 placeholder:text-gray-300 resize-none focus:outline-none focus:ring-2 focus:ring-primary-400/40 ${
                  formErrors.agenda ? 'border-red-300' : 'border-gray-200 focus:border-primary-400'
                }`}
              />
              {formErrors.agenda && <p className="text-xs text-red-500 mt-1">{formErrors.agenda}</p>}
            </div>

            <div className="flex justify-end">
              <Button type="submit">Continue</Button>
            </div>
          </form>
        </Card>
      )}

      {step === 'source' && (
        <>
          {status === 'idle' && (
            <div className="flex items-center justify-between">
              <button
                onClick={() => setStep('details')}
                className="flex items-center gap-1.5 text-sm font-semibold text-gray-500 hover:text-gray-700"
              >
                <ArrowLeft size={15} /> Back to details
              </button>
              <div className="inline-flex p-1 rounded-xl bg-gray-100">
                <button
                  onClick={() => setMode('upload')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    mode === 'upload' ? 'bg-white text-primary-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <UploadCloud size={15} /> Upload File
                </button>
                <button
                  onClick={() => setMode('record')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    mode === 'record' ? 'bg-white text-primary-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Mic size={15} /> Record Audio
                </button>
              </div>
            </div>
          )}

          <Card>
            {status === 'idle' && mode === 'upload' && (
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragOver(true)
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-2xl py-16 cursor-pointer transition-colors ${
                  dragOver ? 'border-primary-400 bg-primary-50/50' : 'border-gray-200 hover:border-primary-300'
                }`}
              >
                <div className="w-14 h-14 rounded-full bg-primary-50 flex items-center justify-center">
                  <UploadCloud size={24} className="text-primary-600" />
                </div>
                <p className="text-sm font-semibold text-gray-800">Drag & drop your file here</p>
                <p className="text-xs text-gray-400">or click to browse from your computer</p>
                <p className="text-xs text-gray-300 mt-2">Accepted formats: MP3, WAV, MP4, M4A</p>
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
              <AudioRecorder onRecordingComplete={handleRecordingComplete} />
            )}

            {status !== 'idle' && file && (
              <div className="space-y-4">
                <div className="flex items-center gap-3 p-4 rounded-xl bg-gray-50 border border-gray-100">
                  <div className="w-10 h-10 rounded-lg bg-primary-50 flex items-center justify-center shrink-0">
                    <FileAudio size={18} className="text-primary-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-gray-800 truncate">{file.name}</p>
                    <p className="text-xs text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  {status === 'done' ? (
                    <CheckCircle2 size={20} className="text-green-500" />
                  ) : (
                    <button onClick={reset} className="text-gray-400 hover:text-gray-600">
                      <X size={18} />
                    </button>
                  )}
                </div>

                <div>
                  <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className="h-full bg-primary-600 rounded-full transition-all duration-150"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    {status === 'done' ? 'Saved to your meeting list.' : `Uploading... ${progress}%`}
                  </p>
                </div>

                {status === 'done' && (
                  <div className="flex gap-3">
                    <Button onClick={startOver} variant="secondary">
                      Create Another Meeting
                    </Button>
                    {savedMeeting && (
                      <Button onClick={() => navigate(`/meetings/${savedMeeting.id}`)}>
                        View Meeting
                      </Button>
                    )}
                  </div>
                )}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
