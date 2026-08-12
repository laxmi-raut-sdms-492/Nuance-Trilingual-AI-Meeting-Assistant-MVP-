/** Must stay aligned with backend ALLOWED_LANGUAGES / LANGUAGE_NAMES. */
export const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English', label: 'EN' },
  { code: 'hi', name: 'Hindi', label: 'HI' },
  { code: 'mr', name: 'Marathi', label: 'MR' },
]

export const SUPPORTED_LANGUAGE_NAMES = SUPPORTED_LANGUAGES.map((l) => l.name).join(', ')

export const LANGUAGE_COLORS = {
  English: '#3B82F6',
  Hindi: '#A855F7',
  Marathi: '#10B981',
}
