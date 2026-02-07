import { NextRequest, NextResponse } from 'next/server'
import { generateFullLessonVideos, estimateCreditsForLesson } from '@/lib/video-orchestrator'
import { heygenClient, type Archetype, type AgeCategory } from '@/lib/heygen-client'
import type { Phase, LessonData } from '@/lib/script-generator'
import { sql } from '@/lib/db'

/**
 * POST /api/heygen/generate-lesson-videos
 * 
 * Generate all 5 phase videos for a specific lesson, age, and archetype.
 * 
 * Body:
 * - dayOfYear: number (1-365)
 * - ageCategory: 'kid' | 'adult' | 'senior'
 * - archetype: one of 12 archetypes
 * - phases?: string[] (optional, defaults to all 5)
 * - testMode?: boolean (optional, default false)
 * - sequential?: boolean (optional, default true)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const {
      dayOfYear,
      ageCategory,
      archetype,
      phases,
      testMode = false,
      sequential = true,
    } = body as {
      dayOfYear: number
      ageCategory: AgeCategory
      archetype: Archetype
      phases?: Phase[]
      testMode?: boolean
      sequential?: boolean
    }

    // Validate required fields
    if (!dayOfYear || !ageCategory || !archetype) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields: dayOfYear, ageCategory, archetype',
      }, { status: 400 })
    }

    // Validate day of year
    if (dayOfYear < 1 || dayOfYear > 365) {
      return NextResponse.json({
        success: false,
        error: 'dayOfYear must be between 1 and 365',
      }, { status: 400 })
    }

    // Check credits
    const estimatedCredits = estimateCreditsForLesson(phases)
    const credits = await heygenClient.getCredits()
    
    if (credits.remaining < estimatedCredits) {
      return NextResponse.json({
        success: false,
        error: `Not enough credits. Need ~${estimatedCredits}, have ${credits.remaining}`,
        credits,
      }, { status: 400 })
    }

    // Fetch lesson data from database
    const lessonRows = await sql`
      SELECT 
        day_of_year,
        topic,
        theme,
        kelly_age,
        hook_script,
        story_script,
        wonder_script,
        action_script,
        wisdom_script,
        wisdom_quote,
        wisdom_author
      FROM lessons 
      WHERE day_of_year = ${dayOfYear}
      LIMIT 1
    `

    if (lessonRows.length === 0) {
      return NextResponse.json({
        success: false,
        error: `Lesson not found for day ${dayOfYear}`,
      }, { status: 404 })
    }

    const lesson = lessonRows[0]

    // Transform to LessonData format
    const lessonData: LessonData = {
      dayOfYear: lesson.day_of_year,
      topic: lesson.topic,
      theme: lesson.theme,
      kellyAge: lesson.kelly_age,
      phases: {
        hook: { question: lesson.hook_script || '' },
        story: { narrative: lesson.story_script || '' },
        wonder: { exploration: lesson.wonder_script || '' },
        action: { activity: lesson.action_script || '' },
        wisdom: {
          quote: lesson.wisdom_quote || '',
          quoteAuthor: lesson.wisdom_author || 'Curious Kelly',
          reflection: lesson.wisdom_script || '',
        },
      },
    }

    console.log(`[GenerateLessonVideos] Starting Day ${dayOfYear} for ${ageCategory}_${archetype}`)

    // Generate all videos
    const result = await generateFullLessonVideos(
      dayOfYear,
      ageCategory,
      archetype,
      lessonData,
      {
        phases,
        testMode,
        storeVideos: true,
        sequential,
      }
    )

    // Calculate actual credits used (rough estimate based on duration)
    const creditsUsed = Math.ceil(result.totalDuration / 10)

    return NextResponse.json({
      success: result.success,
      data: {
        dayOfYear: result.dayOfYear,
        ageCategory: result.ageCategory,
        archetype: result.archetype,
        totalDuration: result.totalDuration,
        creditsUsed,
        phases: Object.fromEntries(
          Object.entries(result.phases).map(([phase, phaseResult]) => [
            phase,
            {
              success: phaseResult.success,
              videoUrl: phaseResult.videoUrl,
              thumbnailUrl: phaseResult.thumbnailUrl,
              duration: phaseResult.duration,
              error: phaseResult.error,
            },
          ])
        ),
      },
    })
  } catch (error) {
    console.error('[GenerateLessonVideos] Error:', error)
    
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Video generation failed',
    }, { status: 500 })
  }
}

/**
 * GET /api/heygen/generate-lesson-videos?day=18
 * 
 * Get info about what would be generated for a lesson.
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const dayOfYear = parseInt(searchParams.get('day') || '1')

  try {
    // Fetch lesson data
    const lessonRows = await sql`
      SELECT topic, theme, kelly_age
      FROM lessons 
      WHERE day_of_year = ${dayOfYear}
      LIMIT 1
    `

    const lesson = lessonRows[0] || null

    // Get credits
    let credits = { remaining: 0, used: 0 }
    try {
      credits = await heygenClient.getCredits()
    } catch {
      // Ignore
    }

    const estimatedCredits = estimateCreditsForLesson()

    return NextResponse.json({
      dayOfYear,
      lesson: lesson ? {
        topic: lesson.topic,
        theme: lesson.theme,
        kellyAge: lesson.kelly_age,
      } : null,
      phases: ['hook', 'story', 'wonder', 'action', 'wisdom'],
      archetypes: [
        'scientist', 'explorer', 'rebel', 'architect',
        'diplomat', 'empath', 'macgyver', 'mystic',
        'provider', 'storyteller', 'strategist', 'survivor',
      ],
      ageCategories: ['kid', 'adult', 'senior'],
      estimatedCredits,
      credits,
      canGenerate: credits.remaining >= estimatedCredits,
      usage: {
        example: `POST /api/heygen/generate-lesson-videos with body: { "dayOfYear": ${dayOfYear}, "ageCategory": "kid", "archetype": "scientist" }`,
      },
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to fetch info',
    }, { status: 500 })
  }
}
