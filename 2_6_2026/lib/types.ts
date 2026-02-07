// ============================================================================
// CURIOUS KELLY - TYPE DEFINITIONS
// ============================================================================

// Lesson Phase Types
export type LessonPhase = 'hook' | 'story' | 'wonder' | 'action' | 'wisdom'

// Age Variants for Kelly (matches live_class_rooms.age_bracket)
export type AgeVariant = 'kid' | 'adult' | 'elder'

// Kelly Archetypes (12 teaching styles with distinctive accessories)
export type KellyArchetype =
  | 'storyteller'  // Gold glasses + peacock feather
  | 'explorer'     // Aviator goggles + leather cap
  | 'architect'    // Tortoiseshell glasses on head
  | 'empath'       // Pink headband + lavender
  | 'rebel'        // Black sunglasses + hoop earrings
  | 'mystic'       // Purple forehead jewel chain
  | 'provider'     // Cream knit headband
  | 'diplomat'     // Navy headband + pearl earrings
  | 'macgyver'     // Shop glasses + red bandana (inventor)
  | 'scientist'    // Lab/safety goggles
  | 'strategist'   // Angular glasses + chess clip
  | 'survivor'     // Military bandana + dog tags

// Live Class Status
export type LiveClassStatus = 'scheduled' | 'starting' | 'live' | 'ended'

// Video Player State
export type VideoState =
  | 'idle'
  | 'loading'
  | 'buffering'
  | 'ready'
  | 'playing'
  | 'paused'
  | 'seeking'
  | 'phase-transition'
  | 'error'
  | 'generating'

// ============================================================================
// DATABASE TYPES (matching Neon schema)
// ============================================================================

export interface User {
  id: string
  name: string | null
  email: string
  emailVerified: boolean
  image: string | null
  role: string | null
  banned: boolean
  banReason: string | null
  banExpires: Date | null
  createdAt: Date
  updatedAt: Date
}

export interface UserPreferences {
  user_id: string
  preferred_avatar_age: number
  preferred_language: string
  preferred_detail_level: string
  preferred_year: number
  days_completed: number
  years_completed: number
  unlocked_years: number[]
  updated_at: Date
}

export interface Lesson {
  id: number
  lesson_id: string
  day_of_year: number
  title: string
  subtitle: string | null
  theme: string
  topic: string
  summary: string | null
  quote: string | null
  quote_author: string | null
  kelly_age: number
  age_range: string | null
  hook_script: string | null // Kelly's engagement question ("Have you ever wondered...")
  hook_fact: string | null // The TRUE fact statement for voting ("Sunlight bounces off...")
  hook_question: string | null // Alias for hook_script (legacy)
  hook_correct_answer: boolean | null // Always true - all hook facts are designed to be TRUE
  teaching_moment: string | null // Full explanation revealed after voting (wonder_script)
  story_script: string | null
  wonder_script: string | null
  action_script: string | null
  wisdom_script: string | null
  created_at: Date
  updated_at: Date
}

export interface LessonPerspective {
  id: number
  day: number
  year: number
  avatar_age: number
  language: string
  title: string
  theme: string | null
  topic: string | null
  context: string | null
  detail_level: string
  phases: {
    hook?: string
    story?: string
    wonder?: string
    action?: string
    wisdom?: string
  }
  asset_ids: string[]
  created_at: Date
  updated_at: Date
}

export interface LiveClassRoom {
  id: string
  day: number
  age_bracket: string
  scheduled_at: Date
  started_at: Date | null
  ended_at: Date | null
  status: LiveClassStatus
  participant_count: number
  max_participants: number
  kelly_present: boolean
  created_at: Date
}

export interface KellyLessonAsset {
  id: string
  day_number: number
  phase: string
  age_group: string
  archetype: string | null
  language: string
  script_text: string | null
  audio_url: string | null
  video_url: string | null
  visual_url: string | null
  video_source: string | null
  video_source_target: string | null
  video_id: string | null
  status: string
  error_message: string | null
  created_at: Date
  updated_at: Date
}

export interface Subscriber {
  id: number
  email: string
  stripe_customer_id: string | null
  subscription_status: string | null
  subscription_tier: string | null
  plan: string | null
  status: string
  is_gift: boolean
  gift_from_email: string | null
  gift_message: string | null
  gift_redeemed_at: Date | null
  has_grow_addon: boolean
  grow_addon_purchased_at: Date | null
  created_at: Date
  updated_at: Date
}

export interface ToneProfile {
  id: number
  archetype: KellyArchetype
  description: string | null
  voice_pace: number
  voice_pitch: number
  voice_energy: number
  motion_intensity: string
  emotional_range: Record<string, unknown>
  gesture_patterns: Record<string, unknown>
  head_props: Record<string, unknown>
  use_cases: Record<string, unknown>
  color_palette: Record<string, unknown>
  created_at: Date
}

export interface Language {
  language_code: string
  language_name: string
  native_name: string | null
  status: string
  priority_tier: number
  speaker_count: number | null
  affiliate_payout_per_lesson: number | null
  launch_date: Date | null
  created_at: Date
}

// ============================================================================
// FRONTEND STATE TYPES
// ============================================================================

export interface LearnerContext {
  // Identity
  userId: string | null
  name: string | null
  displayName: string
  email: string | null
  birthday: Date | null
  isBirthday: boolean
  isAuthenticated: boolean

  // Temporal (always accurate to user's location)
  timezone: string
  locale: string
  now: Date
  dayOfYear: number

  // Formatted strings (ready to display)
  formattedDate: string
  formattedTime: string
  formattedDateTime: string
  timezoneAbbr: string
  greeting: string

  // Preferences
  preferredAge: number
  preferredLanguage: string
  preferredArchetype: KellyArchetype

  // Loading states
  isLoading: boolean
  error: Error | null
  
  // Navigation
  setDayOverride: (day: number | null) => void
}

export interface LiveClassCountdown {
  // Next class data
  class: LiveClassRoom | null

  // Countdown values
  days: number
  hours: number
  minutes: number
  seconds: number

  // State flags
  isStartingSoon: boolean
  isLive: boolean
  hasEnded: boolean
  noUpcoming: boolean

  // Display strings (ALWAYS full format)
  countdownText: string
  fullText: string
  scheduledAtFormatted: string

  // Loading
  isLoading: boolean
  error: Error | null
}

export interface VideoPlayerState {
  state: VideoState
  currentPhase: LessonPhase
  currentTime: number
  duration: number
  buffered: number
  volume: number
  muted: boolean
  captionsEnabled: boolean
  videoUrl: string | null
  fallbackImageUrl: string | null
  error: string | null
}

export interface Caption {
  id: string
  startTime: number
  endTime: number
  text: string
  emphasis?: 'normal' | 'strong' | 'whisper'
}
