// Device fingerprint for anonymous user identification
// Persists across sessions without requiring auth

const FINGERPRINT_KEY = 'kelly_fingerprint'

export function getFingerprint(): string {
  if (typeof window === 'undefined') {
    return 'server-side'
  }
  
  let fingerprint = localStorage.getItem(FINGERPRINT_KEY)
  
  if (!fingerprint) {
    // Generate a new fingerprint based on timestamp + random
    fingerprint = `${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 15)}`
    localStorage.setItem(FINGERPRINT_KEY, fingerprint)
  }
  
  return fingerprint
}

// Generate a fun username based on fingerprint
const ADJECTIVES = [
  'curious', 'eager', 'bright', 'clever', 'wise', 'quick', 'bold', 'calm',
  'happy', 'keen', 'lively', 'merry', 'noble', 'proud', 'swift', 'witty'
]

const NOUNS = [
  'explorer', 'learner', 'thinker', 'seeker', 'dreamer', 'maker', 'builder',
  'reader', 'writer', 'artist', 'scientist', 'student', 'mentor', 'friend'
]

export function generateUsername(fingerprint: string): string {
  // Use fingerprint to deterministically pick words
  const hash = fingerprint.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  const adj = ADJECTIVES[hash % ADJECTIVES.length]
  const noun = NOUNS[(hash * 7) % NOUNS.length]
  const num = (hash % 99) + 1
  
  return `${adj}_${noun}_${num}`
}
