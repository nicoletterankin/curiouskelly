/**
 * Lesson utilities for Lesson of the Day
 * Based on Master Curriculum Thesis - 365 lessons per year
 */

/**
 * Get the day number (1-365) anchored from January 1, 2026.
 * Day 1 = Jan 1 2026, Day 37 = Feb 6 2026, etc.
 * Clamps to 1-365 range.
 */
export function getDayNumber(date: Date = new Date()): number {
  const start = new Date('2026-01-01T00:00:00')
  const now = new Date(date.getFullYear(), date.getMonth(), date.getDate())
  const diffMs = now.getTime() - start.getTime()
  const day = Math.floor(diffMs / (1000 * 60 * 60 * 24)) + 1
  return Math.min(365, Math.max(1, day))
}

/**
 * Get date from day number for current year
 */
export function getDateFromDayNumber(dayNumber: number, year?: number): Date {
  const y = year ?? new Date().getUTCFullYear()
  const date = new Date(Date.UTC(y, 0, 1))
  date.setUTCDate(dayNumber)
  return date
}

/**
 * 5-Phase lesson structure from Master Curriculum Thesis
 * Total: 100 seconds
 */
export const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
export type Phase = typeof PHASES[number]

export const PHASE_DURATIONS: Record<Phase, number> = {
  hook: 15,    // 15 seconds - Grab attention
  story: 30,   // 30 seconds - Narrative illustration
  wonder: 20,  // 20 seconds - Deeper questions
  action: 20,  // 20 seconds - Hands-on activity
  wisdom: 15,  // 15 seconds - Memorable takeaway
}

/**
 * 5 Age buckets from Master Curriculum Thesis
 */
export const AGE_BUCKETS = ['child', 'youth', 'adult', 'mature', 'elder'] as const
export type AgeBucket = typeof AGE_BUCKETS[number]

export const AGE_RANGES: Record<AgeBucket, { min: number; max: number; label: string }> = {
  child: { min: 2, max: 7, label: 'Child (2-7)' },
  youth: { min: 8, max: 14, label: 'Youth (8-14)' },
  adult: { min: 15, max: 44, label: 'Adult (15-44)' },
  mature: { min: 45, max: 64, label: 'Mature (45-64)' },
  elder: { min: 65, max: 120, label: 'Elder (65+)' },
}

/**
 * 12 Jungian Archetypes from Master Curriculum Thesis
 */
export const ARCHETYPES = [
  'explorer', 'sage', 'hero', 'caregiver', 'creator', 'rebel',
  'lover', 'jester', 'everyman', 'ruler', 'magician', 'innocent'
] as const
export type Archetype = typeof ARCHETYPES[number]

/**
 * Get next phase in sequence
 */
export function getNextPhase(currentPhase: Phase): Phase | null {
  const idx = PHASES.indexOf(currentPhase)
  return idx < PHASES.length - 1 ? PHASES[idx + 1] : null
}

/**
 * Get previous phase in sequence
 */
export function getPrevPhase(currentPhase: Phase): Phase | null {
  const idx = PHASES.indexOf(currentPhase)
  return idx > 0 ? PHASES[idx - 1] : null
}

/**
 * Calculate total lesson variants per year
 * 365 days x 5 ages x 12 archetypes x 47 languages = 4,116,600
 */
export const TOTAL_VARIANTS_PER_YEAR = 365 * 5 * 12 * 47
