import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/lesson/[day]
 * 
 * Returns complete lesson data for a specific day.
 * Supports numeric day (1-365) or "today" for current day.
 * Source: Neon database (master_curriculum verified)
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
  
  const rows = await sql`
    SELECT day_of_year, title, topic, theme, 
           hook_fact, hook_correct_answer,
           hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons 
    WHERE day_of_year = ${dayNumber}
    LIMIT 1
  `
  
  if (!rows || rows.length === 0) {
    return NextResponse.json(
      { error: `Lesson not found for day ${dayNumber}` },
      { status: 404 }
    )
  }
  
  const lesson = rows[0]
  
  const response = {
    day: lesson.day_of_year,
    date: getDayDate(lesson.day_of_year as number),
    title: lesson.title,
    theme: lesson.theme || 'General',
    universal_truth: lesson.hook_fact || '',
    ages: ["kid", "adult", "elder"],
    languages: ["en", "es", "fr", "de", "pt", "zh"],
    phases: ["hook", "story", "wonder", "action", "wisdom"],
    content: {
      hook: lesson.hook_script || '',
      story: lesson.story_script || '',
      wonder: lesson.wonder_script || '',
      action: lesson.action_script || '',
      wisdom: lesson.wisdom_script || '',
    },
    canonical_url: `https://www.thedailylesson.com/lesson/${lesson.day_of_year}`,
    provider: {
      name: "Lesson of the Day, PBC",
      url: "https://www.thedailylesson.com"
    },
    license: "CC BY-NC-SA 4.0",
    _generated: new Date().toISOString(),
  }

  return NextResponse.json(response, {
    headers: {
      'Cache-Control': 'public, s-maxage=86400, stale-while-revalidate=604800',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET',
    }
  })
}
