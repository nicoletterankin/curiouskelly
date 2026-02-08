import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/lesson/[day]
 * 
 * Returns complete lesson data for a specific day.
 * Supports numeric day (1-365) or "today" for current day.
 * Source: Neon soft-block database (core_lessons + kellyos_* tables)
 * 
 * DO NOT query the legacy `lessons` table — that's wispy-resonance data.
 */

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

const DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

function getDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0)
  const diff = date.getTime() - start.getTime()
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay)
}

function getDayDate(day: number): string {
  let remaining = day
  for (let i = 0; i < 12; i++) {
    if (remaining <= DAYS_IN_MONTH[i]) {
      return `${MONTHS[i]} ${remaining}`
    }
    remaining -= DAYS_IN_MONTH[i]
  }
  return `Day ${day}`
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ day: string }> }
) {
  const { day: dayParam } = await params
  
  let dayNumber: number
  
  if (dayParam === 'today') {
    dayNumber = getDayOfYear(new Date())
  } else {
    dayNumber = parseInt(dayParam, 10)
  }
  
  if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
    return NextResponse.json(
      { error: 'Invalid day. Must be 1-365 or "today".' },
      { status: 400 }
    )
  }
  
  try {
    // 1. CORE LESSON — from core_lessons (soft-block), NOT legacy `lessons`
    const coreRows = await sql`
      SELECT day_number, title, subject, theme, universal_truth, icon_emoji, marketing_headline
      FROM core_lessons
      WHERE day_number = ${dayNumber}
      LIMIT 1
    `
    
    if (!coreRows || coreRows.length === 0) {
      return NextResponse.json(
        { error: `Lesson not found for day ${dayNumber}`, source: 'core_lessons' },
        { status: 404 }
      )
    }
    
    const lesson = coreRows[0] as Record<string, unknown>
    
    // 2. KELLYOS_LESSONS — phased content (use day_number)
    const phaseRows = await sql`
      SELECT phase, content_text
      FROM kellyos_lessons
      WHERE day_number = ${dayNumber}
        AND language = 'en'
        AND tone = 'mentor'
      ORDER BY phase
    `.catch(() => [])
    
    // Build content map from kellyos_lessons phases
    const phaseNames: Record<number, string> = { 1: 'hook', 2: 'story', 3: 'wonder', 4: 'action', 5: 'wisdom' }
    const content: Record<string, string> = {}
    for (const row of phaseRows as Array<{ phase: number; content_text: string }>) {
      const name = phaseNames[row.phase]
      if (name) content[name] = row.content_text
    }
    
    // 3. KELLYOS_AUDIO — audio URLs (use day_number)
    const audioRows = await sql`
      SELECT phase, audio_url, duration_seconds
      FROM kellyos_audio
      WHERE day_number = ${dayNumber}
        AND language = 'en'
      ORDER BY phase
    `.catch(() => [])
    
    const audioByPhase: Record<string, { audio_url: string; duration_seconds: number }> = {}
    for (const row of audioRows as Array<{ phase: number; audio_url: string; duration_seconds: number }>) {
      const name = phaseNames[row.phase]
      if (name) audioByPhase[name] = { audio_url: row.audio_url, duration_seconds: row.duration_seconds }
    }
    
    // 4. KELLYOS_FACTS — hook fact for True/False (use day_number)
    const factRows = await sql`
      SELECT statement, is_true, explanation
      FROM kellyos_facts
      WHERE day_number = ${dayNumber}
      ORDER BY RANDOM()
      LIMIT 1
    `.catch(() => [])
    
    const fact = (factRows as Array<{ statement: string; is_true: boolean; explanation: string }>)[0] || null
    
    // 5. HEYGEN_VIDEOS — video URLs (uses day_of_year, NOT day_number)
    const videoRows = await sql`
      SELECT phase, video_url, audio_url as heygen_audio_url
      FROM heygen_videos
      WHERE day_of_year = ${dayNumber}
      ORDER BY phase
    `.catch(() => [])
    
    const videoByPhase: Record<string, string> = {}
    for (const row of videoRows as Array<{ phase: string; video_url: string }>) {
      if (row.video_url) videoByPhase[row.phase] = row.video_url
    }
    
    const response = {
      day: dayNumber,
      date: getDayDate(dayNumber),
      title: lesson.title,
      subject: lesson.subject || '',
      theme: lesson.theme || 'General',
      universal_truth: lesson.universal_truth || '',
      icon_emoji: lesson.icon_emoji || '📚',
      ages: ["kid", "adult", "elder"],
      languages: ["en", "es", "fr", "de", "pt", "zh"],
      phases: ["hook", "story", "wonder", "action", "wisdom"],
      content: {
        hook: content.hook || '',
        story: content.story || '',
        wonder: content.wonder || '',
        action: content.action || '',
        wisdom: content.wisdom || '',
      },
      audio: audioByPhase,
      videos: videoByPhase,
      hook_fact: fact?.statement || null,
      hook_correct_answer: fact?.is_true ?? true,
      teaching_moment: fact?.explanation || null,
      canonical_url: `https://www.thedailylesson.com/lesson/${dayNumber}`,
      provider: {
        name: "Lesson of the Day, PBC",
        url: "https://www.thedailylesson.com"
      },
      license: "CC BY-NC-SA 4.0",
      _source: 'core_lessons',
      _generated: new Date().toISOString(),
    }

    return NextResponse.json(response, {
      headers: {
        'Cache-Control': 'public, s-maxage=86400, stale-while-revalidate=604800',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
      }
    })
  } catch (error) {
    console.error('[lesson/day] Unexpected error:', error)
    return NextResponse.json(
      { error: 'Failed to load lesson', day: dayNumber },
      { status: 500 }
    )
  }
}
