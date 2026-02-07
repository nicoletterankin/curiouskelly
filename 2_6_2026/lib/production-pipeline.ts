// ============================================================================
// PRODUCTION PIPELINE
// Unified orchestrator for full 365-day, all-variant video coverage
// 
// Coordinates: Lessons DB -> Scripts -> Audio (ElevenLabs) -> kelly_lesson_assets -> Video (Fal.ai lip-sync)
// 
// Variant Matrix (per day):
//   3 ages (kid, adult, senior)
//   x 12 archetypes (storyteller, explorer, scientist, architect, strategist, diplomat, mystic, rebel, macgyver, empath, provider, survivor)
//   x 5 phases (hook, story, wonder, action, wisdom)
//   = 180 videos per day
//   = 65,700 videos for 365 days (English only)
// ============================================================================

import type { NeonQueryFunction } from '@neondatabase/serverless'

// ============================================================================
// CONSTANTS - Single source of truth for all pipeline dimensions
// ============================================================================

export const PIPELINE_AGES = ['kid', 'adult', 'senior'] as const
export type PipelineAge = typeof PIPELINE_AGES[number]

export const PIPELINE_ARCHETYPES = [
  'storyteller', 'explorer', 'scientist', 'architect',
  'strategist', 'diplomat', 'mystic', 'rebel',
  'macgyver', 'empath', 'provider', 'survivor',
] as const
export type PipelineArchetype = typeof PIPELINE_ARCHETYPES[number]

export const PIPELINE_PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
export type PipelinePhase = typeof PIPELINE_PHASES[number]

export const SLOTS_PER_DAY = PIPELINE_AGES.length * PIPELINE_ARCHETYPES.length * PIPELINE_PHASES.length // 180
export const TOTAL_DAYS = 365
export const TOTAL_SLOTS = SLOTS_PER_DAY * TOTAL_DAYS // 65,700

// Phase -> DB column mapping for lessons table
export const PHASE_SCRIPT_COLUMNS: Record<PipelinePhase, string> = {
  hook: 'hook_script',
  story: 'story_script',
  wonder: 'wonder_script',
  action: 'action_script',
  wisdom: 'wisdom_script',
}

// ============================================================================
// TYPES
// ============================================================================

export interface PipelineSlot {
  day_number: number
  phase: PipelinePhase
  age_group: PipelineAge
  archetype: PipelineArchetype
  language: string
}

export interface SlotStatus extends PipelineSlot {
  has_script: boolean
  has_audio: boolean
  has_video: boolean
  audio_url: string | null
  video_url: string | null
  video_source: string | null
  status: string | null
  kelly_asset_id: number | null
}

export interface DayCoverage {
  day_number: number
  title: string | null
  has_lesson: boolean
  total_slots: number
  slots_seeded: number
  slots_with_script: boolean // lesson scripts exist
  slots_with_audio: number
  slots_with_video: number
  completion_pct: number
}

export interface PipelineSummary {
  total_days: number
  days_with_lessons: number
  days_missing_lessons: number[]
  total_slots: number
  slots_seeded: number
  slots_with_audio: number
  slots_with_video: number
  audio_pct: number
  video_pct: number
  by_age: Record<PipelineAge, { audio: number; video: number; total: number }>
  by_archetype: Record<string, { audio: number; video: number; total: number }>
  by_phase: Record<PipelinePhase, { audio: number; video: number; total: number }>
}

// ============================================================================
// PIPELINE STATUS QUERIES
// ============================================================================

/**
 * Get full pipeline summary across all 365 days
 */
export async function getPipelineSummary(sql: NeonQueryFunction<false, false>): Promise<PipelineSummary> {
  // Count lessons in DB
  const lessonCount = await sql`
    SELECT COUNT(DISTINCT day_of_year) as count FROM lessons
  `
  const daysWithLessons = parseInt(lessonCount[0]?.count as string) || 0

  // Find which days are missing
  const existingDays = await sql`
    SELECT DISTINCT day_of_year FROM lessons ORDER BY day_of_year
  `
  const existingDaySet = new Set((existingDays as Array<{ day_of_year: number }>).map(d => d.day_of_year))
  const daysMissingLessons: number[] = []
  for (let d = 1; d <= 365; d++) {
    if (!existingDaySet.has(d)) daysMissingLessons.push(d)
  }

  // Count kelly_lesson_assets rows and their states
  const assetStats = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
    WHERE language = 'en'
  `
  const stats = assetStats[0] || { total: 0, with_audio: 0, with_video: 0 }
  const slotsSeeded = parseInt(stats.total as string) || 0
  const slotsWithAudio = parseInt(stats.with_audio as string) || 0
  const slotsWithVideo = parseInt(stats.with_video as string) || 0

  // By age group
  const byAgeRaw = await sql`
    SELECT 
      age_group,
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
    WHERE language = 'en'
    GROUP BY age_group
  `
  const byAge = {} as Record<PipelineAge, { audio: number; video: number; total: number }>
  for (const row of byAgeRaw as Array<{ age_group: string; total: string; with_audio: string; with_video: string }>) {
    const mapped = mapToBaseAge(row.age_group)
    if (!byAge[mapped]) byAge[mapped] = { audio: 0, video: 0, total: 0 }
    byAge[mapped].total += parseInt(row.total) || 0
    byAge[mapped].audio += parseInt(row.with_audio) || 0
    byAge[mapped].video += parseInt(row.with_video) || 0
  }

  // By archetype
  const byArchRaw = await sql`
    SELECT 
      archetype,
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
    WHERE language = 'en'
    GROUP BY archetype
  `
  const byArchetype = {} as Record<string, { audio: number; video: number; total: number }>
  for (const row of byArchRaw as Array<{ archetype: string; total: string; with_audio: string; with_video: string }>) {
    byArchetype[row.archetype || 'unknown'] = {
      total: parseInt(row.total) || 0,
      audio: parseInt(row.with_audio) || 0,
      video: parseInt(row.with_video) || 0,
    }
  }

  // By phase
  const byPhaseRaw = await sql`
    SELECT 
      phase,
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
    WHERE language = 'en'
    GROUP BY phase
  `
  const byPhase = {} as Record<PipelinePhase, { audio: number; video: number; total: number }>
  for (const row of byPhaseRaw as Array<{ phase: string; total: string; with_audio: string; with_video: string }>) {
    byPhase[row.phase as PipelinePhase] = {
      total: parseInt(row.total) || 0,
      audio: parseInt(row.with_audio) || 0,
      video: parseInt(row.with_video) || 0,
    }
  }

  return {
    total_days: TOTAL_DAYS,
    days_with_lessons: daysWithLessons,
    days_missing_lessons: daysMissingLessons,
    total_slots: TOTAL_SLOTS,
    slots_seeded: slotsSeeded,
    slots_with_audio: slotsWithAudio,
    slots_with_video: slotsWithVideo,
    audio_pct: TOTAL_SLOTS > 0 ? (slotsWithAudio / TOTAL_SLOTS) * 100 : 0,
    video_pct: TOTAL_SLOTS > 0 ? (slotsWithVideo / TOTAL_SLOTS) * 100 : 0,
    by_age: byAge,
    by_archetype: byArchetype,
    by_phase: byPhase,
  }
}

/**
 * Get coverage for a specific day 
 */
export async function getDayCoverage(sql: NeonQueryFunction<false, false>, dayNumber: number): Promise<DayCoverage> {
  // Check if lesson exists
  const lesson = await sql`
    SELECT day_of_year, title, hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons WHERE day_of_year = ${dayNumber} LIMIT 1
  `
  const hasLesson = lesson.length > 0
  const hasScripts = hasLesson && !!(lesson[0].hook_script || lesson[0].story_script)

  // Count kelly_lesson_assets for this day
  const assetStats = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
    WHERE day_number = ${dayNumber} AND language = 'en'
  `
  const stats = assetStats[0] || { total: 0, with_audio: 0, with_video: 0 }
  const seeded = parseInt(stats.total as string) || 0
  const withAudio = parseInt(stats.with_audio as string) || 0
  const withVideo = parseInt(stats.with_video as string) || 0

  return {
    day_number: dayNumber,
    title: hasLesson ? (lesson[0].title as string) : null,
    has_lesson: hasLesson,
    total_slots: SLOTS_PER_DAY,
    slots_seeded: seeded,
    slots_with_script: hasScripts,
    slots_with_audio: withAudio,
    slots_with_video: withVideo,
    completion_pct: SLOTS_PER_DAY > 0 ? (withVideo / SLOTS_PER_DAY) * 100 : 0,
  }
}

/**
 * Get coverage for a range of days
 */
export async function getDayRangeCoverage(
  sql: NeonQueryFunction<false, false>,
  startDay: number,
  endDay: number
): Promise<DayCoverage[]> {
  // Get all lessons in range
  const lessons = await sql`
    SELECT day_of_year, title, 
           hook_script IS NOT NULL as has_hook,
           story_script IS NOT NULL as has_story
    FROM lessons 
    WHERE day_of_year >= ${startDay} AND day_of_year <= ${endDay}
    ORDER BY day_of_year
  `
  const lessonMap = new Map<number, { title: string; has_scripts: boolean }>()
  for (const l of lessons as Array<{ day_of_year: number; title: string; has_hook: boolean; has_story: boolean }>) {
    lessonMap.set(l.day_of_year, { title: l.title, has_scripts: l.has_hook || l.has_story })
  }

  // Get kelly_lesson_assets stats grouped by day
  const assetStats = await sql`
    SELECT 
      day_number,
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets
    WHERE day_number >= ${startDay} AND day_number <= ${endDay} AND language = 'en'
    GROUP BY day_number
    ORDER BY day_number
  `
  const assetMap = new Map<number, { total: number; with_audio: number; with_video: number }>()
  for (const a of assetStats as Array<{ day_number: number; total: string; with_audio: string; with_video: string }>) {
    assetMap.set(a.day_number, {
      total: parseInt(a.total) || 0,
      with_audio: parseInt(a.with_audio) || 0,
      with_video: parseInt(a.with_video) || 0,
    })
  }

  // Build results for each day in range
  const results: DayCoverage[] = []
  for (let d = startDay; d <= endDay; d++) {
    const lesson = lessonMap.get(d)
    const assets = assetMap.get(d) || { total: 0, with_audio: 0, with_video: 0 }
    results.push({
      day_number: d,
      title: lesson?.title || null,
      has_lesson: !!lesson,
      total_slots: SLOTS_PER_DAY,
      slots_seeded: assets.total,
      slots_with_script: !!lesson?.has_scripts,
      slots_with_audio: assets.with_audio,
      slots_with_video: assets.with_video,
      completion_pct: SLOTS_PER_DAY > 0 ? (assets.with_video / SLOTS_PER_DAY) * 100 : 0,
    })
  }

  return results
}

// ============================================================================
// SLOT SEEDING
// Ensures all 180 rows exist in kelly_lesson_assets for a given day
// ============================================================================

/**
 * Seed all 180 slots for a single day into kelly_lesson_assets.
 * Idempotent - skips rows that already exist.
 * Returns count of newly inserted rows.
 */
export async function seedDaySlots(
  sql: NeonQueryFunction<false, false>,
  dayNumber: number,
  language: string = 'en'
): Promise<{ inserted: number; existing: number }> {
  // Check which slots already exist
  const existing = await sql`
    SELECT phase, age_group, archetype
    FROM kelly_lesson_assets
    WHERE day_number = ${dayNumber} AND language = ${language}
  `
  const existingSet = new Set(
    (existing as Array<{ phase: string; age_group: string; archetype: string }>)
      .map(r => `${r.phase}|${r.age_group}|${r.archetype}`)
  )

  // Get the lesson script content to populate script_text
  const lesson = await sql`
    SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons WHERE day_of_year = ${dayNumber} LIMIT 1
  `
  const lessonData = lesson[0] as Record<string, string | null> | undefined

  let inserted = 0
  const toInsert: Array<{
    day: number; phase: string; age: string; archetype: string; lang: string; script: string | null
  }> = []

  for (const phase of PIPELINE_PHASES) {
    const scriptCol = PHASE_SCRIPT_COLUMNS[phase]
    const scriptText = lessonData?.[scriptCol] || null

    for (const age of PIPELINE_AGES) {
      for (const archetype of PIPELINE_ARCHETYPES) {
        const key = `${phase}|${age}|${archetype}`
        if (!existingSet.has(key)) {
          toInsert.push({
            day: dayNumber, phase, age, archetype, lang: language, script: scriptText,
          })
        }
      }
    }
  }

  // Batch insert in chunks of 50
  for (let i = 0; i < toInsert.length; i += 50) {
    const chunk = toInsert.slice(i, i + 50)
    for (const row of chunk) {
      await sql`
        INSERT INTO kelly_lesson_assets (day_number, phase, age_group, archetype, language, script_text, status, created_at, updated_at)
        VALUES (${row.day}, ${row.phase}, ${row.age}, ${row.archetype}, ${row.lang}, ${row.script}, 'pending', NOW(), NOW())
        ON CONFLICT DO NOTHING
      `
      inserted++
    }
  }

  return { inserted, existing: existingSet.size }
}

/**
 * Seed slots for a range of days. Only seeds days that have lessons.
 */
export async function seedDayRange(
  sql: NeonQueryFunction<false, false>,
  startDay: number,
  endDay: number,
  language: string = 'en'
): Promise<{ days_processed: number; total_inserted: number; total_existing: number }> {
  // Get days that have lessons
  const lessonDays = await sql`
    SELECT day_of_year FROM lessons
    WHERE day_of_year >= ${startDay} AND day_of_year <= ${endDay}
    ORDER BY day_of_year
  `

  let totalInserted = 0
  let totalExisting = 0

  for (const row of lessonDays as Array<{ day_of_year: number }>) {
    const { inserted, existing } = await seedDaySlots(sql, row.day_of_year, language)
    totalInserted += inserted
    totalExisting += existing
  }

  return {
    days_processed: lessonDays.length,
    total_inserted: totalInserted,
    total_existing: totalExisting,
  }
}

// ============================================================================
// AUDIO POPULATION
// Copies audio from lesson_audio table to kelly_lesson_assets,
// or generates new audio via ElevenLabs and writes directly to kelly_lesson_assets
// ============================================================================

/**
 * Find kelly_lesson_assets rows that need audio (seeded but no audio_url).
 */
export async function findSlotsNeedingAudio(
  sql: NeonQueryFunction<false, false>,
  options: { dayStart?: number; dayEnd?: number; age?: PipelineAge; limit?: number } = {}
): Promise<Array<{ id: string; day_number: number; phase: string; age_group: string; archetype: string }>> {
  const { dayStart = 1, dayEnd = 365, age, limit = 100 } = options

  if (age) {
    return sql`
      SELECT id, day_number, phase, age_group, archetype
      FROM kelly_lesson_assets
      WHERE audio_url IS NULL
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
        AND age_group = ${age}
        AND language = 'en'
      ORDER BY day_number, phase, age_group, archetype
      LIMIT ${limit}
    ` as Promise<Array<{ id: string; day_number: number; phase: string; age_group: string; archetype: string }>>
  }

  return sql`
    SELECT id, day_number, phase, age_group, archetype
    FROM kelly_lesson_assets
    WHERE audio_url IS NULL
      AND day_number >= ${dayStart}
      AND day_number <= ${dayEnd}
      AND language = 'en'
    ORDER BY day_number, phase, age_group, archetype
    LIMIT ${limit}
  ` as Promise<Array<{ id: string; day_number: number; phase: string; age_group: string; archetype: string }>>
}

/**
 * Find kelly_lesson_assets rows that need video (have audio but no video).
 */
export async function findSlotsNeedingVideo(
  sql: NeonQueryFunction<false, false>,
  options: { dayStart?: number; dayEnd?: number; age?: PipelineAge; limit?: number } = {}
): Promise<Array<{ id: string; day_number: number; phase: string; age_group: string; archetype: string; audio_url: string }>> {
  const { dayStart = 1, dayEnd = 365, age, limit = 100 } = options

  if (age) {
    return sql`
      SELECT id, day_number, phase, age_group, archetype, audio_url
      FROM kelly_lesson_assets
      WHERE audio_url IS NOT NULL
        AND video_url IS NULL
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
        AND age_group = ${age}
        AND language = 'en'
      ORDER BY day_number, phase, age_group, archetype
      LIMIT ${limit}
    ` as Promise<Array<{ id: string; day_number: number; phase: string; age_group: string; archetype: string; audio_url: string }>>
  }

  return sql`
    SELECT id, day_number, phase, age_group, archetype, audio_url
    FROM kelly_lesson_assets
    WHERE audio_url IS NOT NULL
      AND video_url IS NULL
      AND day_number >= ${dayStart}
      AND day_number <= ${dayEnd}
      AND language = 'en'
    ORDER BY day_number, phase, age_group, archetype
    LIMIT ${limit}
  ` as Promise<Array<{ id: string; day_number: number; phase: string; age_group: string; archetype: string; audio_url: string }>>
}

// ============================================================================
// UTILITIES
// ============================================================================

/** Map any DB age group string to the 3 base video ages */
export function mapToBaseAge(ageGroup: string): PipelineAge {
  switch (ageGroup) {
    case 'toddler':
    case 'kid':
    case 'child':
      return 'kid'
    case 'preteen':
    case 'youngAdult':
    case 'adult':
    case 'middleAge':
      return 'adult'
    case 'senior':
    case 'elder':
      return 'senior'
    default:
      return 'adult'
  }
}

/** Generate all 180 slot keys for a single day */
export function generateDaySlotKeys(dayNumber: number): PipelineSlot[] {
  const slots: PipelineSlot[] = []
  for (const phase of PIPELINE_PHASES) {
    for (const age of PIPELINE_AGES) {
      for (const archetype of PIPELINE_ARCHETYPES) {
        slots.push({ day_number: dayNumber, phase, age_group: age, archetype, language: 'en' })
      }
    }
  }
  return slots
}
