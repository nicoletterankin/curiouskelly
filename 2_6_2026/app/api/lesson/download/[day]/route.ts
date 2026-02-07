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
  quote?: string
  quoteAuthor?: string
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
    // Get base lesson data
    // NOTE: Only query columns that are VERIFIED to exist in the lessons table
    // subtitle, *_visual_url, content_version may not exist - use defensive querying
    const lessonData = await sql`
      SELECT 
        day_of_year,
        title,
        topic,
        theme,
        quote,
        quote_author,
        hook_script,
        story_script,
        wonder_script,
        action_script,
        wisdom_script
      FROM lessons
      WHERE day_of_year = ${dayNumber}
      LIMIT 1
    `

    if (!lessonData || lessonData.length === 0) {
      return NextResponse.json({ error: 'Lesson not found' }, { status: 404 })
    }

    const lesson = lessonData[0] as Record<string, unknown>

    // Map age group to database variants
    const ageVariants = getAgeVariants(ageGroup)

    // Get all audio URLs for this lesson
    const audioAssets = await sql`
      SELECT phase, audio_url, script_text
      FROM kelly_lesson_assets
      WHERE day_number = ${dayNumber}
        AND language = ${language}
        AND audio_url IS NOT NULL
        AND (age_group = ${ageVariants[0]} OR age_group = ${ageVariants[1]} OR age_group = ${ageVariants[2] || ageVariants[1]})
      ORDER BY 
        CASE WHEN age_group = ${ageVariants[0]} THEN 0 
             WHEN age_group = ${ageVariants[1]} THEN 1 
             ELSE 2 END
    `.catch(() => [])

    // Build audio map
    const audioMap = new Map<string, { audioUrl: string; scriptText?: string }>()
    for (const asset of audioAssets as Array<{ phase: string; audio_url: string; script_text?: string }>) {
      if (!audioMap.has(asset.phase)) {
        audioMap.set(asset.phase, { 
          audioUrl: asset.audio_url, 
          scriptText: asset.script_text 
        })
      }
    }

    // Try to get personalized scripts from lesson_perspectives
    const perspectiveAge = ['child', 'teen'].includes(ageGroup) ? 'kid' : 
                           ageGroup === 'middleAge' ? 'adult' : ageGroup
    
    const perspectiveData = await sql`
      SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lesson_perspectives
      WHERE day_number = ${dayNumber}
        AND age_group = ${perspectiveAge}
        AND archetype = ${archetype}
        AND language = ${language}
      LIMIT 1
    `.catch(() => [])

    const perspective = perspectiveData?.[0] as Record<string, unknown> | undefined

    // Build phase content
    const phases: Record<string, PhaseContent> = {}
    
    for (const phase of PHASES) {
      const scriptCol = `${phase}_script`
      const audioInfo = audioMap.get(phase)

      phases[phase] = {
        // Priority: perspective script > audio script > base lesson script
        script: (perspective?.[scriptCol] as string) || 
                audioInfo?.scriptText || 
                (lesson[scriptCol] as string) || 
                getDefaultScript(phase),
        audioUrl: audioInfo?.audioUrl || null,
        visualUrl: null, // visual_url columns don't exist in lessons table
      }
    }

    const offlineLesson: OfflineLesson = {
      dayNumber,
      title: lesson.title as string,
      topic: lesson.topic as string | undefined,
      theme: lesson.theme as string | undefined,
      quote: lesson.quote as string | undefined,
      quoteAuthor: lesson.quote_author as string | undefined,
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
