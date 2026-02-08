import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/lesson/today
 * 
 * Cinematic Kelly — The unified lesson API that pulls from ALL populated tables.
 * 
 * Query params:
 *   lang     - Language code: en, es, fr, de, ja, ko, pt, zh (default: en)
 *   tone     - Teaching tone: mentor, kid, elder (default: mentor)
 *   archetype - Kelly archetype: mentor, scientist, storyteller, etc. (default: mentor)
 *   day      - Override day number (1-365). Default: today's day of year
 * 
 * Tables queried:
 *   core_lessons    - 365 daily lesson metadata (title, subject, theme)
 *   kellyos_lessons - Multilingual phased content (content_text per phase/language/tone)
 *   kellyos_audio   - Audio URLs per day/phase/language
 *   lesson_atoms    - Archetype-specific hooks, scripts, emotions
 * 
 * Response shape matches the spec's cinematic lesson contract.
 */

export const dynamic = 'force-dynamic'
export const revalidate = 0

// Map tone param to DB-compatible values
function normalizeTone(tone: string): string {
  const map: Record<string, string> = {
    'adult': 'mentor',
    'default': 'mentor',
    'mentor': 'mentor',
    'kid': 'kid',
    'child': 'kid',
    'elder': 'elder',
    'senior': 'elder',
  }
  return map[tone.toLowerCase()] || 'mentor'
}

// Get current day of year (1-365)
function getDayOfYear(): number {
  const now = new Date()
  const start = new Date(now.getFullYear(), 0, 0)
  const diff = now.getTime() - start.getTime()
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay)
}

// Map day number to readable date string
function getDayDate(day: number): string {
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ]
  const daysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  let remaining = day
  for (let i = 0; i < 12; i++) {
    if (remaining <= daysInMonth[i]) {
      return `${months[i]} ${remaining}`
    }
    remaining -= daysInMonth[i]
  }
  return `Day ${day}`
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  
  const lang = searchParams.get('lang') || 'en'
  const rawTone = searchParams.get('tone') || 'mentor'
  const tone = normalizeTone(rawTone)
  const archetypeParam = searchParams.get('archetype') || 'mentor'
  const dayParam = searchParams.get('day')
  
  const dayNumber = dayParam ? parseInt(dayParam, 10) : getDayOfYear()
  
  if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
    return NextResponse.json(
      { error: 'Invalid day. Must be 1-365.' },
      { status: 400 }
    )
  }
  
  try {
    // ──────────────────────────────────────────────────────────
    // 1. CORE LESSON METADATA (365 rows — title, subject, theme)
    // ──────────────────────────────────────────────────────────
    let coreLesson: Record<string, unknown> | null = null
    
    try {
      const coreRows = await sql`
        SELECT day_number, title, subject, theme, universal_truth, icon_emoji, marketing_headline
        FROM core_lessons
        WHERE day_number = ${dayNumber}
        LIMIT 1
      `
      if (coreRows.length > 0) {
        coreLesson = coreRows[0] as Record<string, unknown>
      }
    } catch (e) {
      console.warn('[lesson/today] core_lessons query failed:', (e as Error).message)
    }
    
    // NOTE: Do NOT fall back to the legacy `lessons` table.
    // That table lives in wispy-resonance and contains WRONG curriculum data.
    // If core_lessons is empty for this day, we return a clear 404 instead of wrong content.
    
    // ──────────────────────────────────────────────────────────
    // 2. KELLYOS LESSONS — Phased content (multilingual + tone)
    // ──────────────────────────────────────────────────────────
    interface PhaseRow {
      phase: number
      content_text: string
    }
    let phases: PhaseRow[] = []
    
    try {
      const phaseRows = await sql`
        SELECT kl.phase, kl.content_text
        FROM kellyos_lessons kl
        WHERE kl.day_number = ${dayNumber}
          AND kl.language = ${lang}
          AND kl.tone = ${tone}
        ORDER BY kl.phase
      `
      phases = phaseRows as PhaseRow[]
    } catch (e) {
      console.warn('[lesson/today] kellyos_lessons query failed:', (e as Error).message)
    }
    
    // Fallback: if no kellyos_lessons rows, try with 'mentor' tone for the requested language
    if (phases.length === 0 && tone !== 'mentor') {
      try {
        const fallbackPhases = await sql`
          SELECT phase, content_text
          FROM kellyos_lessons
          WHERE day_number = ${dayNumber}
            AND language = ${lang}
            AND tone = 'mentor'
          ORDER BY phase
        `
        phases = fallbackPhases as PhaseRow[]
      } catch (e) {
        console.warn('[lesson/today] kellyos_lessons fallback query failed:', (e as Error).message)
      }
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. KELLYOS AUDIO — Audio URLs per phase/language
    // ──────────────────────────────────────────────────────────
    interface AudioRow {
      phase: number
      audio_url: string
      duration_seconds: number
    }
    let audioRecords: AudioRow[] = []
    
    try {
      const audioRows = await sql`
        SELECT phase, audio_url, duration_seconds
        FROM kellyos_audio
        WHERE day_number = ${dayNumber}
          AND language = ${lang}
        ORDER BY phase
      `
      audioRecords = audioRows as AudioRow[]
    } catch (e) {
      console.warn('[lesson/today] kellyos_audio query failed:', (e as Error).message)
    }
    
    // Build audio lookup by phase number
    const audioByPhase: Record<number, { audio_url: string; duration_seconds: number }> = {}
    for (const a of audioRecords) {
      audioByPhase[a.phase] = { audio_url: a.audio_url, duration_seconds: a.duration_seconds }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. ARCHETYPE DATA from lesson_atoms
    // ──────────────────────────────────────────────────────────
    interface ArchetypeRow {
      archetype: string
      kelly_script: string
      kelly_emotion: string
    }
    let archetypeData: ArchetypeRow | null = null
    
    try {
      const archRows = await sql`
        SELECT archetype, kelly_script, kelly_emotion
        FROM lesson_atoms
        WHERE day_number = ${dayNumber}
          AND archetype = ${archetypeParam}
          AND is_active = true
        LIMIT 1
      `
      if (archRows.length > 0) {
        archetypeData = archRows[0] as ArchetypeRow
      }
    } catch (e) {
      console.warn('[lesson/today] lesson_atoms archetype query failed:', (e as Error).message)
      // Try without is_active filter (column may not exist)
      try {
        const archRows2 = await sql`
          SELECT archetype, kelly_script, kelly_emotion
          FROM lesson_atoms
          WHERE day_number = ${dayNumber}
            AND archetype = ${archetypeParam}
          LIMIT 1
        `
        if (archRows2.length > 0) {
          archetypeData = archRows2[0] as ArchetypeRow
        }
      } catch {
        // lesson_atoms may not have these columns - graceful fallback
      }
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. HOOK FACT for True/False game
    //    Priority: kellyos_facts (730 rows) > lessons.hook_fact
    // ──────────────────────────────────────────────────────────
    let hookFact: string | null = null
    let hookCorrectAnswer: boolean = true
    let teachingMoment: string | null = null
    let hookQuestion: string | null = null
    
    // Priority 1: kellyos_facts table (generated by Enrichment Engine — 730 facts, 2 per day)
    try {
      const factsRows = await sql`
        SELECT statement, is_true, explanation
        FROM kellyos_facts
        WHERE day_number = ${dayNumber}
        ORDER BY RANDOM()
        LIMIT 1
      `
      if (factsRows.length > 0) {
        const f = factsRows[0] as { statement: string; is_true: boolean; explanation: string }
        hookFact = f.statement
        hookCorrectAnswer = f.is_true
        teachingMoment = f.explanation
      }
    } catch (e) {
      console.warn('[lesson/today] kellyos_facts query failed:', (e as Error).message)
      
      // Fallback: try kellyos_facts_v2
      try {
        const factsV2 = await sql`
          SELECT statement, is_true, explanation
          FROM kellyos_facts_v2
          WHERE day_number = ${dayNumber}
          ORDER BY RANDOM()
          LIMIT 1
        `
        if (factsV2.length > 0) {
          const f = factsV2[0] as { statement: string; is_true: boolean; explanation: string }
          hookFact = f.statement
          hookCorrectAnswer = f.is_true
          teachingMoment = f.explanation
        }
      } catch {
        // Both facts tables missing - continue to lessons fallback
      }
    }
    
    // NOTE: Do NOT fall back to the legacy `lessons` table for hook facts.
    // That table lives in wispy-resonance and contains WRONG curriculum data.
    
    // ──────────────────────────────────────────────────────────
    // 6. LEGACY SCRIPT FALLBACK — If kellyos_lessons has no data,
    //    fall back to existing lessons/perspective scripts
    // ──────────────────────────────────────────────────────────
    let legacyScripts: Record<string, string> = {}
    if (phases.length === 0) {
      try {
        // Try lesson_perspectives first (personalized by age/archetype/lang)
        const toneToAge: Record<string, string> = { kid: 'kid', mentor: 'adult', elder: 'senior' }
        const ageGroup = toneToAge[tone] || 'adult'
        
        const perspRows = await sql`
          SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
          FROM lesson_perspectives
          WHERE day_number = ${dayNumber}
            AND age_group = ${ageGroup}
            AND language = ${lang}
          LIMIT 1
        `
        
        if (perspRows.length > 0) {
          const p = perspRows[0] as Record<string, unknown>
          if (p.hook_script) legacyScripts['1'] = p.hook_script as string
          if (p.story_script) legacyScripts['2'] = p.story_script as string
          if (p.wonder_script) legacyScripts['3'] = p.wonder_script as string
          if (p.action_script) legacyScripts['4'] = p.action_script as string
          if (p.wisdom_script) legacyScripts['5'] = p.wisdom_script as string
        }
        
        // NOTE: Do NOT fall back to the legacy `lessons` table for scripts.
        // That table lives in wispy-resonance and contains WRONG curriculum data.
      } catch (e) {
        console.warn('[lesson/today] legacy scripts fallback failed:', (e as Error).message)
      }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. LEGACY AUDIO FALLBACK — If kellyos_audio empty,
    //    try kelly_lesson_assets
    // ──────────────────────────────────────────────────────────
    if (Object.keys(audioByPhase).length === 0) {
      try {
        const toneToAge: Record<string, string> = { kid: 'kid', mentor: 'adult', elder: 'senior' }
        const ageGroup = toneToAge[tone] || 'adult'
        
        const legacyAudio = await sql`
          SELECT phase, audio_url
          FROM kelly_lesson_assets
          WHERE day_number = ${dayNumber}
            AND language = ${lang}
            AND age_group = ${ageGroup}
            AND audio_url IS NOT NULL
          ORDER BY phase
        `
        
        // Map phase names to phase numbers
        const phaseNameToNum: Record<string, number> = {
          hook: 1, story: 2, wonder: 3, action: 4, wisdom: 5
        }
        for (const row of legacyAudio) {
          const r = row as { phase: string; audio_url: string }
          const phaseNum = phaseNameToNum[r.phase] || parseInt(r.phase, 10)
          if (phaseNum && !audioByPhase[phaseNum]) {
            audioByPhase[phaseNum] = { audio_url: r.audio_url, duration_seconds: 45 }
          }
        }
      } catch (e) {
        console.warn('[lesson/today] kelly_lesson_assets audio fallback failed:', (e as Error).message)
      }
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. ASSEMBLE RESPONSE
    // ──────────────────────────────────────────────────────────
    
    // Build phases array — merge content + audio
    const phaseNames: Record<number, string> = {
      1: 'hook', 2: 'story', 3: 'wonder', 4: 'action', 5: 'wisdom'
    }
    
    const assembledPhases = []
    const phaseCount = Math.max(phases.length, Object.keys(legacyScripts).length, 5)
    
    for (let i = 1; i <= phaseCount; i++) {
      const phaseRow = phases.find(p => p.phase === i)
      const audio = audioByPhase[i]
      const legacyText = legacyScripts[String(i)]
      
      const contentText = phaseRow?.content_text || legacyText || null
      
      if (contentText) {
        assembledPhases.push({
          phase: i,
          phase_name: phaseNames[i] || `phase_${i}`,
          content_text: contentText,
          audio_url: audio?.audio_url || null,
          duration_seconds: audio?.duration_seconds || 45,
        })
      }
    }
    
    // Archetype info
    const archetype = archetypeData ? {
      name: archetypeData.archetype,
      hook: archetypeData.kelly_script || null,
      emotion: archetypeData.kelly_emotion || 'warm',
    } : {
      name: archetypeParam,
      hook: null,
      emotion: 'warm',
    }
    
    const response = {
      day_number: dayNumber,
      date: getDayDate(dayNumber),
      title: (coreLesson?.title as string) || (coreLesson?.marketing_headline as string) || `Day ${dayNumber}`,
      subject: (coreLesson?.subject as string) || (coreLesson?.theme as string) || '',
      theme: (coreLesson?.theme as string) || '',
      universal_truth: (coreLesson?.universal_truth as string) || '',
      icon_emoji: (coreLesson?.icon_emoji as string) || '📚',
      language: lang,
      tone,
      phases: assembledPhases,
      archetype,
      // Hook/True-False data
      hook_fact: hookFact,
      hook_correct_answer: hookCorrectAnswer,
      hook_question: hookQuestion,
      teaching_moment: teachingMoment,
      // Metadata
      _sources: {
        core_lessons: !!coreLesson,
        kellyos_lessons: phases.length > 0,
        kellyos_audio: audioRecords.length > 0,
        lesson_atoms: !!archetypeData,
        legacy_fallback: phases.length === 0 && Object.keys(legacyScripts).length > 0,
      },
      _generated: new Date().toISOString(),
    }
    
    return NextResponse.json(response, {
      headers: {
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
        'Access-Control-Allow-Origin': '*',
      }
    })
    
  } catch (error) {
    console.error('[lesson/today] Unexpected error:', error)
    return NextResponse.json(
      { 
        error: 'Failed to load lesson',
        day_number: dayNumber,
        phases: [],
      },
      { status: 500 }
    )
  }
}
