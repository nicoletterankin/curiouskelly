import { NextRequest, NextResponse } from 'next/server'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

// CRITICAL: These are CONTAMINATED topics from the API export
// They should NEVER be in the Learn Track
const CONTAMINATED_TOPICS = [
  'Why Popcorn Pops',
  'How Dolphins Sleep',
  'Why Cheese Has Holes',
  'How Archer Fish Hunt',
  'Why Wet Stones Look Prettier',
  'How Snowflakes Are All Different',
  'Why Bruises Change Color',
  'How Bees Make Hexagons',
  'Why Crabs Walk Sideways',
  'How Holograms Work',
  'Why Yawning Is Contagious',
  'How Diamonds Form',
  'Why Fevers Help Fight Illness',
  'How Fireworks Get Their Colors',
  'Why Old Photographs Smell',
  'How Platypuses Sense Electricity',
  'Why Some Beaches Have Pink Sand',
  'How Lie Detectors Try to Work',
  'Why Calendars Have 12 Months',
  'How Geysers Erupt',
]

// Known CORRECT Learn Track topics from ANTIGRAVITY-ALIGNMENT.md
const VERIFIED_LEARN_TOPICS: Record<number, string> = {
  1: 'Why Do We Dream?',
  2: 'How Do Birds Fly?',
  18: 'What Happens When You Rest',
  365: 'What Will Tomorrow Bring?',
}

export async function GET(request: NextRequest) {
  try {
    // Get all lessons and check for contamination
    const sql = getSql()
    const lessons = await sql`
      SELECT 
        day_of_year, 
        title, 
        track, 
        verified, 
        curriculum_source 
      FROM lessons 
      ORDER BY day_of_year
    `

    const contaminated: Array<{ day: number; title: string; reason: string }> = []
    const verified: Array<{ day: number; title: string }> = []
    const unknown: Array<{ day: number; title: string }> = []

    for (const lesson of lessons) {
      const title = lesson.title

      // Check if it's a known contaminated topic
      if (CONTAMINATED_TOPICS.some(t => title?.toLowerCase().includes(t.toLowerCase()))) {
        contaminated.push({
          day: lesson.day_of_year,
          title,
          reason: 'Matches known placeholder content from API export',
        })
        continue
      }

      // Check if it matches a verified Learn Track topic
      const expectedTitle = VERIFIED_LEARN_TOPICS[lesson.day_of_year]
      if (expectedTitle) {
        if (title?.toLowerCase() === expectedTitle.toLowerCase()) {
          verified.push({ day: lesson.day_of_year, title })
        } else {
          contaminated.push({
            day: lesson.day_of_year,
            title,
            reason: `Expected "${expectedTitle}" but got "${title}"`,
          })
        }
        continue
      }

      // Unknown - needs manual verification
      unknown.push({ day: lesson.day_of_year, title })
    }

    return NextResponse.json({
      summary: {
        total: lessons.length,
        contaminated: contaminated.length,
        verified: verified.length,
        unknown: unknown.length,
        status: contaminated.length > 0 ? 'CONTAMINATED' : 'CLEAN',
      },
      contaminated,
      verified,
      // Only show first 10 unknown for brevity
      unknown: unknown.slice(0, 10),
      message:
        contaminated.length > 0
          ? 'DATABASE IS CONTAMINATED. Do not disable maintenance mode until curriculum is fixed.'
          : 'No known contamination detected, but manual verification still needed.',
      protection: {
        track_column: 'Added',
        verified_column: 'Added',
        curriculum_source_column: 'Added',
        doc: '/docs/CRITICAL-TRACK-PROTECTION.md',
      },
    })
  } catch (error) {
    console.error('Validation error:', error)
    return NextResponse.json(
      { error: 'Failed to validate curriculum', details: String(error) },
      { status: 500 }
    )
  }
}

// POST: Mark a lesson as verified or contaminated
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { day, verified, curriculum_source, correct_title } = body

    if (!day) {
      return NextResponse.json({ error: 'day is required' }, { status: 400 })
    }

    const sql = getSql()

    // If providing correct_title, update the lesson
    if (correct_title) {
      await sql`
        UPDATE lessons 
        SET title = ${correct_title},
            verified = true,
            curriculum_source = 'curated_20yr'
        WHERE day_of_year = ${day}
      `
      return NextResponse.json({
        success: true,
        message: `Day ${day} updated to "${correct_title}" and marked as verified`,
      })
    }

    // Otherwise just update verification status
    await sql`
      UPDATE lessons 
      SET verified = ${verified ?? false},
          curriculum_source = ${curriculum_source ?? 'unknown'}
      WHERE day_of_year = ${day}
    `

    return NextResponse.json({
      success: true,
      message: `Day ${day} updated: verified=${verified}, source=${curriculum_source}`,
    })
  } catch (error) {
    console.error('Update error:', error)
    return NextResponse.json({ error: 'Failed to update lesson', details: String(error) }, { status: 500 })
  }
}
