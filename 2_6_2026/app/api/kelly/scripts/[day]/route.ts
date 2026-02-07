/**
 * GET /api/kelly/scripts/[day]
 *
 * Returns Kelly's variant scripts from the kelly_scripts table.
 *
 * Query params:
 *   ?variant=kid|adult|elder  – filter to a single variant (default: all 3)
 *   ?phase=hook|story|...     – filter to a single phase (default: all 5)
 *
 * Examples:
 *   /api/kelly/scripts/37                   – all variants, all phases
 *   /api/kelly/scripts/37?variant=kid       – kid variant only
 *   /api/kelly/scripts/37?variant=adult&phase=hook – adult hook only
 *   /api/kelly/scripts/today                – today's scripts
 */

import { NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

function getDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0)
  const diff = date.getTime() - start.getTime()
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay)
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ day: string }> }
) {
  const { day: dayParam } = await params
  const { searchParams } = new URL(request.url)
  const variantFilter = searchParams.get('variant')
  const phaseFilter = searchParams.get('phase')

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

  // Validate variant/phase if provided
  const validVariants = ['kid', 'adult', 'elder']
  const validPhases = ['hook', 'story', 'wonder', 'action', 'wisdom']

  if (variantFilter && !validVariants.includes(variantFilter)) {
    return NextResponse.json(
      { error: `Invalid variant. Must be one of: ${validVariants.join(', ')}` },
      { status: 400 }
    )
  }

  if (phaseFilter && !validPhases.includes(phaseFilter)) {
    return NextResponse.json(
      { error: `Invalid phase. Must be one of: ${validPhases.join(', ')}` },
      { status: 400 }
    )
  }

  try {
    const sql = getSql()

    // Build query based on filters
    let rows: Record<string, unknown>[]

    if (variantFilter && phaseFilter) {
      rows = await sql`
        SELECT day_number, variant, phase, script_text, word_count,
               audio_generated, video_generated, generated_at, generation_model
        FROM kelly_scripts
        WHERE day_number = ${dayNumber}
          AND variant = ${variantFilter}
          AND phase = ${phaseFilter}
        ORDER BY variant, phase
      `
    } else if (variantFilter) {
      rows = await sql`
        SELECT day_number, variant, phase, script_text, word_count,
               audio_generated, video_generated, generated_at, generation_model
        FROM kelly_scripts
        WHERE day_number = ${dayNumber}
          AND variant = ${variantFilter}
        ORDER BY phase
      `
    } else if (phaseFilter) {
      rows = await sql`
        SELECT day_number, variant, phase, script_text, word_count,
               audio_generated, video_generated, generated_at, generation_model
        FROM kelly_scripts
        WHERE day_number = ${dayNumber}
          AND phase = ${phaseFilter}
        ORDER BY variant
      `
    } else {
      rows = await sql`
        SELECT day_number, variant, phase, script_text, word_count,
               audio_generated, video_generated, generated_at, generation_model
        FROM kelly_scripts
        WHERE day_number = ${dayNumber}
        ORDER BY variant, phase
      `
    }

    if (!rows || rows.length === 0) {
      return NextResponse.json(
        {
          error: `No kelly_scripts found for day ${dayNumber}${variantFilter ? ` variant=${variantFilter}` : ''}${phaseFilter ? ` phase=${phaseFilter}` : ''}`,
          day: dayNumber,
          hint: `Generate scripts via POST /api/admin/kelly-scripts/generate { day: ${dayNumber} }`,
        },
        { status: 404 }
      )
    }

    // Get lesson metadata
    const lessonMeta = await sql`
      SELECT title, topic, theme FROM lessons WHERE day_of_year = ${dayNumber} LIMIT 1
    `

    // Structure response: group by variant
    const structured: Record<
      string,
      Record<string, { text: string; word_count: number }>
    > = {}
    const meta: Record<
      string,
      { generated_at: unknown; model: unknown; audio: boolean; video: boolean }
    > = {}

    for (const row of rows) {
      const v = row.variant as string
      const p = row.phase as string

      if (!structured[v]) structured[v] = {}
      structured[v][p] = {
        text: row.script_text as string,
        word_count: row.word_count as number,
      }

      // Track generation metadata per variant
      if (!meta[v]) {
        meta[v] = {
          generated_at: row.generated_at,
          model: row.generation_model,
          audio: true,
          video: true,
        }
      }
      if (!row.audio_generated) meta[v].audio = false
      if (!row.video_generated) meta[v].video = false
    }

    const lesson = lessonMeta[0] as Record<string, unknown> | undefined

    return NextResponse.json(
      {
        day: dayNumber,
        title: lesson?.title || null,
        topic: lesson?.topic || null,
        theme: lesson?.theme || null,
        variants: Object.keys(structured),
        phases: validPhases,
        scripts: structured,
        pipeline_status: meta,
        total_scripts: rows.length,
      },
      {
        headers: {
          'Cache-Control':
            'public, s-maxage=3600, stale-while-revalidate=86400',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET',
        },
      }
    )
  } catch (error) {
    console.error('[kelly/scripts] GET error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
