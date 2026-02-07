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
    // Get all lessons in range
    const lessonsData = await sql`
      SELECT 
        day_of_year,
        title,
        topic,
        hook_script,
        story_script,
        wonder_script,
        action_script,
        wisdom_script,
        hook_visual_url,
        wonder_visual_url,
        content_version
      FROM lessons
      WHERE day_of_year >= ${start} AND day_of_year <= ${actualEnd}
      ORDER BY day_of_year
    `

    if (!lessonsData || lessonsData.length === 0) {
      return NextResponse.json({ error: 'No lessons found' }, { status: 404 })
    }

    // Map age group to database variants
    const ageVariants = getAgeVariants(ageGroup)
    const dayNumbers = (lessonsData as Array<{ day_of_year: number }>).map(l => l.day_of_year)

    // Get all audio URLs for these lessons in one query
    const audioAssets = await sql`
      SELECT day_number, phase, audio_url
      FROM kelly_lesson_assets
      WHERE day_number = ANY(${dayNumbers})
        AND language = ${language}
        AND audio_url IS NOT NULL
        AND (age_group = ${ageVariants[0]} OR age_group = ${ageVariants[1]} OR age_group = ${ageVariants[2] || ageVariants[1]})
    `.catch(() => [])

    // Build audio map: day -> phase -> audioUrl
    const audioMap = new Map<number, Map<string, string>>()
    for (const asset of audioAssets as Array<{ day_number: number; phase: string; audio_url: string }>) {
      if (!audioMap.has(asset.day_number)) {
        audioMap.set(asset.day_number, new Map())
      }
      const dayMap = audioMap.get(asset.day_number)!
      if (!dayMap.has(asset.phase)) {
        dayMap.set(asset.phase, asset.audio_url)
      }
    }

    // Build compact lesson objects
    const lessons: OfflineLessonCompact[] = []
    let totalWithAudio = 0

    for (const lesson of lessonsData as Array<Record<string, unknown>>) {
      const dayNum = lesson.day_of_year as number
      const dayAudio = audioMap.get(dayNum)

      const phases: Record<string, PhaseContent> = {}
      let hasAnyAudio = false

      for (const phase of PHASES) {
        const scriptCol = `${phase}_script`
        const visualCol = `${phase}_visual_url`
        const audioUrl = dayAudio?.get(phase) || null
        
        if (audioUrl) hasAnyAudio = true

        phases[phase] = {
          script: (lesson[scriptCol] as string) || getDefaultScript(phase),
          audioUrl,
          visualUrl: (lesson[visualCol] as string) || null,
        }
      }

      if (hasAnyAudio) totalWithAudio++

      lessons.push({
        day: dayNum,
        title: lesson.title as string,
        topic: lesson.topic as string | undefined,
        phases,
        version: (lesson.content_version as number) || 1,
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
