/**
 * CALENDAR API - Fetches lesson data for calendar grid
 * 
 * GET /api/calendar
 * Returns all 365 lessons with titles, themes, and completion status
 * 
 * GET /api/calendar?month=1
 * Returns lessons for a specific month
 * 
 * GET /api/calendar?day=45
 * Returns a single day's lesson details
 */

import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// Cache lesson data for 1 hour
const CACHE_DURATION = 3600

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = searchParams.get('day')
  const month = searchParams.get('month')
  const userId = searchParams.get('userId')

  try {
    // Single day - full details
    if (day) {
      const dayNum = parseInt(day)
      
      const lessons = await sql`
        SELECT 
          day_of_year,
          title,
          topic,
          theme,
          summary,
          quote,
          quote_author,
          hook_script,
          story_script,
          wonder_script,
          action_script,
          wisdom_script
        FROM lessons
        WHERE day_of_year = ${dayNum}
        LIMIT 1
      `

      if (lessons.length === 0) {
        return NextResponse.json({ error: 'Lesson not found' }, { status: 404 })
      }

      // Get audio assets if available (kelly_lesson_assets has audio_url, not video_url)
      const assets = await sql`
        SELECT 
          phase,
          audio_url,
          visual_url
        FROM kelly_lesson_assets
        WHERE day_number = ${dayNum} AND language = 'en'
        LIMIT 5
      `

      // Get user progress if userId provided
      let progress = null
      if (userId) {
        const userProgress = await sql`
          SELECT phase, completed_at, time_spent_seconds
          FROM user_progress
          WHERE user_id = ${userId} AND day_of_year = ${dayNum}
        `
        progress = userProgress
      }

      return NextResponse.json({
        lesson: lessons[0],
        assets: assets.reduce((acc: Record<string, any>, a: any) => {
          acc[a.phase] = { audio: a.audio_url, visual: a.visual_url }
          return acc
        }, {}),
        progress,
      })
    }

    // Month view - get lessons for specific month
    if (month) {
      const monthNum = parseInt(month)
      // Calculate day range for month (approximate)
      const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
      const startDay = daysInMonth.slice(0, monthNum - 1).reduce((a, b) => a + b, 0) + 1
      const endDay = startDay + daysInMonth[monthNum - 1] - 1

      const lessons = await sql`
        SELECT day_of_year, title, topic, theme
        FROM lessons
        WHERE day_of_year >= ${startDay} AND day_of_year <= ${endDay}
        ORDER BY day_of_year
      `

      // Get completion status if userId
      let completedDays: number[] = []
      if (userId) {
        const completed = await sql`
          SELECT DISTINCT day_of_year 
          FROM user_progress 
          WHERE user_id = ${userId} AND completed_at IS NOT NULL
        `
        completedDays = completed.map((c: { day_of_year: number }) => c.day_of_year)
      }

      return NextResponse.json({
        month: monthNum,
        startDay,
        endDay,
        lessons,
        completedDays,
      })
    }

    // Full year - lightweight data for calendar grid
    const lessons = await sql`
      SELECT 
        day_of_year,
        title,
        topic,
        theme
      FROM lessons
      ORDER BY day_of_year
    `

    // Get themes for color coding
    const themes = await sql`
      SELECT DISTINCT theme, COUNT(*)::int as count
      FROM lessons
      GROUP BY theme
      ORDER BY count DESC
    `

    // Get completion status if userId
    let completedDays: number[] = []
    if (userId) {
      const completed = await sql`
        SELECT DISTINCT day_of_year 
        FROM user_progress 
        WHERE user_id = ${userId} AND completed_at IS NOT NULL
      `
      completedDays = completed.map((c: { day_of_year: number }) => c.day_of_year)
    }

    // Transform into calendar grid format
    const calendarData = lessons.map((l: any) => ({
      day: l.day_of_year,
      title: l.title,
      topic: l.topic,
      theme: l.theme,
      completed: completedDays.includes(l.day_of_year),
    }))

    return NextResponse.json({
      year: 2026,
      totalLessons: lessons.length,
      themes: themes.map((t: any) => ({ name: t.theme, count: t.count })),
      completedCount: completedDays.length,
      lessons: calendarData,
    }, {
      headers: {
        'Cache-Control': `public, s-maxage=${CACHE_DURATION}, stale-while-revalidate`,
      },
    })

  } catch (error) {
    console.error('[calendar] Error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch calendar data' },
      { status: 500 }
    )
  }
}
