/**
 * LESSON RANGE DOWNLOAD ENDPOINT
 * 
 * Returns multiple lessons for offline storage (e.g., a week or month).
 * Optimized for batch downloading with progress tracking.
 * 
 * GET /api/lesson/download/range?start=1&end=7&age=adult&language=en
 * 
 * Returns: { lessons: OfflineLesson[], meta: { total, withAudio, size } }
 */

import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

interface PhaseContent {
  script: string
  audioUrl: string | null
  visualUrl: string | null
}

interface OfflineLessonCompact {
  day: number
  title: string
  topic?: string
  phases: Record<string, PhaseContent>
  version: number
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  
  const start = parseInt(searchParams.get('start') || '1', 10)
  const end = parseInt(searchParams.get('end') || '7', 10)
  const ageGroup = searchParams.get('age') || 'adult'
  const language = searchParams.get('language') || 'en'

  // Validate range
  if (start < 1 || end > 365 || start > end) {
    return NextResponse.json({ error: 'Invalid range (1-365)' }, { status: 400 })
  }

  // Limit to 30 lessons per request to avoid timeouts
  const maxLessons = 30
  const actualEnd = Math.min(end, start + maxLessons - 1)

  try {
    // Get all lessons in range from core_lessons (soft-block), NOT legacy `lessons`
    const lessonsData = await sql`
      SELECT 
        day_number,
        title,
        subject as topic,
        theme,
        icon_emoji
      FROM core_lessons
      WHERE day_number >= ${start} AND day_number <= ${actualEnd}
      ORDER BY day_number
    `

    if (!lessonsData || lessonsData.length === 0) {
      return NextResponse.json({ error: 'No lessons found' }, { status: 404 })
    }

    const dayNumbers = (lessonsData as Array<{ day_number: number }>).map(l => l.day_number)

    // Get kellyos_lessons content for all days in range
    const kellyosContent = await sql`
      SELECT day_number, phase, content_text
      FROM kellyos_lessons
      WHERE day_number = ANY(${dayNumbers})
        AND language = ${language}
        AND tone = 'mentor'
      ORDER BY day_number, phase
    `.catch(() => [])

    // Build content map: day -> phase_num -> content_text
    const phaseNames: Record<number, string> = { 1: 'hook', 2: 'story', 3: 'wonder', 4: 'action', 5: 'wisdom' }
    const contentMap = new Map<number, Map<string, string>>()
    for (const row of kellyosContent as Array<{ day_number: number; phase: number; content_text: string }>) {
      if (!contentMap.has(row.day_number)) {
        contentMap.set(row.day_number, new Map())
      }
      const name = phaseNames[row.phase]
      if (name) contentMap.get(row.day_number)!.set(name, row.content_text)
    }

    // Get kellyos_audio for all days in range
    const audioAssets = await sql`
      SELECT day_number, phase, audio_url, duration_seconds
      FROM kellyos_audio
      WHERE day_number = ANY(${dayNumbers})
        AND language = ${language}
      ORDER BY day_number, phase
    `.catch(() => [])

    // Build audio map: day -> phase_name -> audioUrl
    const audioMap = new Map<number, Map<string, string>>()
    for (const asset of audioAssets as Array<{ day_number: number; phase: number; audio_url: string }>) {
      if (!audioMap.has(asset.day_number)) {
        audioMap.set(asset.day_number, new Map())
      }
      const name = phaseNames[asset.phase]
      if (name && !audioMap.get(asset.day_number)!.has(name)) {
        audioMap.get(asset.day_number)!.set(name, asset.audio_url)
      }
    }

    // Build compact lesson objects
    const lessons: OfflineLessonCompact[] = []
    let totalWithAudio = 0

    for (const lesson of lessonsData as Array<Record<string, unknown>>) {
      const dayNum = lesson.day_number as number
      const dayContent = contentMap.get(dayNum)
      const dayAudio = audioMap.get(dayNum)

      const phases: Record<string, PhaseContent> = {}
      let hasAnyAudio = false

      for (const phase of PHASES) {
        const audioUrl = dayAudio?.get(phase) || null
        
        if (audioUrl) hasAnyAudio = true

        phases[phase] = {
          script: dayContent?.get(phase) || getDefaultScript(phase),
          audioUrl,
          visualUrl: null,
        }
      }

      if (hasAnyAudio) totalWithAudio++

      lessons.push({
        day: dayNum,
        title: lesson.title as string,
        topic: lesson.topic as string | undefined,
        phases,
        version: 1,
      })
    }

    // Calculate approximate size
    const dataStr = JSON.stringify(lessons)
    const estimatedSize = dataStr.length

    return NextResponse.json({
      lessons,
      meta: {
        rangeStart: start,
        rangeEnd: actualEnd,
        total: lessons.length,
        withAudio: totalWithAudio,
        withoutAudio: lessons.length - totalWithAudio,
        estimatedSizeBytes: estimatedSize,
        estimatedSizeKB: Math.round(estimatedSize / 1024),
        ageGroup,
        language,
        hasMore: end > actualEnd,
        nextStart: actualEnd + 1,
      }
    })

  } catch (error) {
    console.error('[lesson/download/range] Error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch lessons' },
      { status: 500 }
    )
  }
}

function getAgeVariants(ageGroup: string): string[] {
  const mapping: Record<string, string[]> = {
    'child': ['toddler', 'preteen', 'kid'],
    'youth': ['preteen', 'youngAdult', 'teen'],
    'adult': ['youngAdult', 'middleAge', 'adult'],
    'mature': ['middleAge', 'senior', 'adult'],
    'elder': ['senior', 'elder', 'middleAge'],
  }
  return mapping[ageGroup] || ['adult', 'youngAdult', 'middleAge']
}

function getDefaultScript(phase: string): string {
  const scripts: Record<string, string> = {
    hook: "Welcome to today's lesson!",
    story: "Let me share this story with you.",
    wonder: "What questions does this raise?",
    action: "Let's try something together!",
    wisdom: "Here's what we've learned.",
  }
  return scripts[phase] || scripts.hook
}
