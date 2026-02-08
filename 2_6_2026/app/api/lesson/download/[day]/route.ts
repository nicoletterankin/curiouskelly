/**
 * LESSON DOWNLOAD ENDPOINT
 * 
 * Returns a complete lesson bundle for offline storage.
 * Includes all phases, scripts, audio URLs, and visual URLs.
 * 
 * GET /api/lesson/download/19?age=adult&language=en&archetype=explorer
 * 
 * Returns: { lesson: OfflineLesson, size: number }
 */

import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

interface PhaseContent {
  script: string
  audioUrl: string | null
  visualUrl: string | null
  duration?: number
}

interface OfflineLesson {
  dayNumber: number
  title: string
  topic?: string
  theme?: string
  phases: Record<string, PhaseContent>
  downloadedAt: string
  version: number
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ day: string }> }
) {
  const { day } = await params
  const dayNumber = parseInt(day, 10)

  if (!dayNumber || dayNumber < 1 || dayNumber > 365) {
    return NextResponse.json({ error: 'Invalid day (1-365)' }, { status: 400 })
  }

  const { searchParams } = new URL(request.url)
  const ageGroup = searchParams.get('age') || 'adult'
  const language = searchParams.get('language') || 'en'
  const archetype = searchParams.get('archetype') || 'explorer'

  try {
    // Get base lesson data from core_lessons (soft-block), NOT legacy `lessons`
    const lessonData = await sql`
      SELECT 
        day_number,
        title,
        subject as topic,
        theme,
        universal_truth,
        icon_emoji
      FROM core_lessons
      WHERE day_number = ${dayNumber}
      LIMIT 1
    `

    if (!lessonData || lessonData.length === 0) {
      return NextResponse.json({ error: 'Lesson not found', source: 'core_lessons' }, { status: 404 })
    }

    const lesson = lessonData[0] as Record<string, unknown>

    // Get kellyos_lessons content (use day_number)
    const kellyosContent = await sql`
      SELECT phase, content_text
      FROM kellyos_lessons
      WHERE day_number = ${dayNumber}
        AND language = ${language}
        AND tone = 'mentor'
      ORDER BY phase
    `.catch(() => [])

    const phaseNames: Record<number, string> = { 1: 'hook', 2: 'story', 3: 'wonder', 4: 'action', 5: 'wisdom' }
    const contentByPhase = new Map<string, string>()
    for (const row of kellyosContent as Array<{ phase: number; content_text: string }>) {
      const name = phaseNames[row.phase]
      if (name) contentByPhase.set(name, row.content_text)
    }

    // Get kellyos_audio for this lesson (use day_number)
    const audioAssets = await sql`
      SELECT phase, audio_url, duration_seconds
      FROM kellyos_audio
      WHERE day_number = ${dayNumber}
        AND language = ${language}
      ORDER BY phase
    `.catch(() => [])

    // Build audio map by phase name
    const audioMap = new Map<string, { audioUrl: string; duration: number }>()
    for (const asset of audioAssets as Array<{ phase: number; audio_url: string; duration_seconds: number }>) {
      const name = phaseNames[asset.phase]
      if (name && !audioMap.has(name)) {
        audioMap.set(name, { audioUrl: asset.audio_url, duration: asset.duration_seconds })
      }
    }

    // Build phase content
    const phases: Record<string, PhaseContent> = {}
    
    for (const phase of PHASES) {
      const audioInfo = audioMap.get(phase)

      phases[phase] = {
        script: contentByPhase.get(phase) || getDefaultScript(phase),
        audioUrl: audioInfo?.audioUrl || null,
        visualUrl: null,
        duration: audioInfo?.duration,
      }
    }

    const offlineLesson: OfflineLesson = {
      dayNumber,
      title: lesson.title as string,
      topic: lesson.topic as string | undefined,
      theme: lesson.theme as string | undefined,
      phases,
      downloadedAt: new Date().toISOString(),
      version: 1,
    }

    // Estimate size (rough calculation for UI display)
    const estimatedSize = JSON.stringify(offlineLesson).length

    return NextResponse.json({
      lesson: offlineLesson,
      meta: {
        ageGroup,
        language,
        archetype,
        hasAudio: Object.values(phases).filter(p => p.audioUrl).length,
        hasVisuals: Object.values(phases).filter(p => p.visualUrl).length,
        estimatedSize,
      }
    })

  } catch (error) {
    console.error('[lesson/download] Error:', error)
    return NextResponse.json(
      { error: 'Failed to prepare lesson for download' },
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
    hook: "Welcome to today's lesson! Let's spark your curiosity.",
    story: "Every great discovery starts with a story.",
    wonder: "Now let's dive deeper and explore together.",
    action: "Time to put what we've learned into practice!",
    wisdom: "What have we discovered today? Let's reflect.",
  }
  return scripts[phase] || scripts.hook
}
