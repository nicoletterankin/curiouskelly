/**
 * BATCH AUDIO GENERATION ENDPOINT
 * 
 * Generates audio for multiple lessons/phases in a single request.
 * Designed for filling coverage gaps across the 365-day curriculum.
 * 
 * POST /api/audio/batch
 * Body: { dayStart: 1, dayEnd: 30, phases: ['hook', 'story'], ageGroup: 'adult', language: 'en' }
 * 
 * Returns: { queued: number, existing: number, errors: number }
 */

import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
type Phase = typeof PHASES[number]

const AGE_GROUPS = ['child', 'youth', 'adult', 'mature', 'elder'] as const
type AgeGroup = typeof AGE_GROUPS[number]

// Map age group to database values
const AGE_DB_MAP: Record<AgeGroup, string[]> = {
  child: ['toddler', 'preteen', 'kid'],
  youth: ['preteen', 'youngAdult', 'teen'],
  adult: ['youngAdult', 'middleAge', 'adult'],
  mature: ['middleAge', 'senior', 'adult'],
  elder: ['senior', 'elder'],
}

interface BatchRequest {
  dayStart?: number
  dayEnd?: number
  days?: number[]
  phases?: Phase[]
  ageGroup?: AgeGroup
  language?: string
  archetype?: string
  dryRun?: boolean
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as BatchRequest
    
    const {
      dayStart = 1,
      dayEnd = 30,
      days,
      phases = ['hook', 'story', 'wisdom'],
      ageGroup = 'adult',
      language = 'en',
      archetype = 'explorer',
      dryRun = false,
    } = body

    // Determine which days to process
    const targetDays = days || Array.from(
      { length: dayEnd - dayStart + 1 }, 
      (_, i) => dayStart + i
    )

    // Validate
    if (targetDays.some(d => d < 1 || d > 365)) {
      return NextResponse.json({ error: 'Days must be 1-365' }, { status: 400 })
    }

    if (phases.some(p => !PHASES.includes(p))) {
      return NextResponse.json({ error: 'Invalid phase' }, { status: 400 })
    }

    // Get age variants for database query
    const ageVariants = AGE_DB_MAP[ageGroup] || ['adult']

    // Check existing coverage
    const existingAssets = await sql`
      SELECT day_number, phase
      FROM kelly_lesson_assets
      WHERE day_number = ANY(${targetDays})
        AND phase = ANY(${phases})
        AND language = ${language}
        AND audio_url IS NOT NULL
        AND (age_group = ${ageVariants[0]} OR age_group = ${ageVariants[1]} OR age_group = ${ageVariants[2] || ageVariants[1]})
    `.catch(() => [])

    // Build set of existing day-phase combos
    const existingSet = new Set(
      (existingAssets as Array<{ day_number: number; phase: string }>)
        .map(a => `${a.day_number}-${a.phase}`)
    )

    // Find gaps
    const gaps: Array<{ day: number; phase: string }> = []
    for (const day of targetDays) {
      for (const phase of phases) {
        if (!existingSet.has(`${day}-${phase}`)) {
          gaps.push({ day, phase })
        }
      }
    }

    if (dryRun) {
      return NextResponse.json({
        dryRun: true,
        targetDays: targetDays.length,
        targetPhases: phases.length,
        totalCombinations: targetDays.length * phases.length,
        existing: existingSet.size,
        gaps: gaps.length,
        gapList: gaps.slice(0, 50), // First 50 gaps for inspection
        estimatedCredits: gaps.length * 500, // ~500 chars per script avg
        estimatedCost: `$${(gaps.length * 500 * 0.00003).toFixed(2)}`, // ElevenLabs pricing
      })
    }

    // Queue generation jobs
    let queued = 0
    let errors = 0
    const results: Array<{ day: number; phase: string; status: string }> = []

    // Process in batches of 10 to avoid overwhelming APIs
    for (let i = 0; i < gaps.length; i += 10) {
      const batch = gaps.slice(i, i + 10)
      
      await Promise.all(
        batch.map(async ({ day, phase }) => {
          try {
            // Call the TTS endpoint to generate audio
            const ttsUrl = new URL('/api/audio/tts', request.url)
            ttsUrl.searchParams.set('day', day.toString())
            ttsUrl.searchParams.set('phase', phase)
            ttsUrl.searchParams.set('age', ageGroup)
            ttsUrl.searchParams.set('language', language)
            ttsUrl.searchParams.set('archetype', archetype)

            const response = await fetch(ttsUrl.toString(), {
              headers: { 'x-internal-request': 'true' },
            })

            if (response.ok) {
              queued++
              results.push({ day, phase, status: 'queued' })
            } else {
              errors++
              results.push({ day, phase, status: 'error' })
            }
          } catch {
            errors++
            results.push({ day, phase, status: 'error' })
          }
        })
      )

      // Small delay between batches to respect rate limits
      if (i + 10 < gaps.length) {
        await new Promise(r => setTimeout(r, 1000))
      }
    }

    return NextResponse.json({
      success: true,
      summary: {
        targetDays: targetDays.length,
        targetPhases: phases.length,
        existing: existingSet.size,
        queued,
        errors,
      },
      results: results.slice(0, 100), // First 100 results
    })

  } catch (error) {
    console.error('[audio/batch] Error:', error)
    return NextResponse.json(
      { error: 'Batch processing failed', details: String(error) },
      { status: 500 }
    )
  }
}

// GET endpoint for checking coverage status
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const language = searchParams.get('language') || 'en'

  try {
    // Count total coverage by phase
    const coverage = await sql`
      SELECT 
        phase,
        COUNT(DISTINCT day_number) as days_covered,
        COUNT(*) as total_assets
      FROM kelly_lesson_assets
      WHERE language = ${language}
        AND audio_url IS NOT NULL
      GROUP BY phase
      ORDER BY phase
    `

    // Count by age group
    const byAge = await sql`
      SELECT 
        age_group,
        COUNT(DISTINCT day_number) as days_covered
      FROM kelly_lesson_assets
      WHERE language = ${language}
        AND audio_url IS NOT NULL
      GROUP BY age_group
      ORDER BY days_covered DESC
    `

    // Get lessons without any audio
    const missingAudio = await sql`
      SELECT l.day_of_year, l.title
      FROM lessons l
      LEFT JOIN kelly_lesson_assets kla 
        ON l.day_of_year = kla.day_number 
        AND kla.audio_url IS NOT NULL
        AND kla.language = ${language}
      WHERE kla.id IS NULL
      ORDER BY l.day_of_year
      LIMIT 50
    `

    return NextResponse.json({
      language,
      coverage: {
        byPhase: coverage,
        byAge: byAge,
        totalLessons: 365,
        lessonsWithAudio: new Set((coverage as Array<{ phase: string; days_covered: number }>).flatMap(c => c.days_covered)).size,
      },
      missingAudio: {
        count: (missingAudio as Array<unknown>).length,
        sample: missingAudio,
      },
    })

  } catch (error) {
    console.error('[audio/batch] Coverage check error:', error)
    return NextResponse.json(
      { error: 'Coverage check failed' },
      { status: 500 }
    )
  }
}
