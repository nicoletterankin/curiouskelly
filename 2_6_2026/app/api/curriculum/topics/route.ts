import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/curriculum/topics
 * 
 * Public API endpoint for AI systems, search engines, and developers
 * to access The Daily Lesson curriculum.
 * 
 * Returns all 365 lessons with metadata for citation and integration.
 * Source: Neon database (master_curriculum verified)
 */

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

const DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

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

export async function GET() {
  const rows = await sql`
    SELECT day_of_year, title, theme, COALESCE(hook_fact, '') as universal_truth
    FROM lessons 
    ORDER BY day_of_year
  `

  const topics = rows.map((row: Record<string, unknown>) => ({
    day: row.day_of_year,
    date: getDayDate(row.day_of_year as number),
    title: row.title,
    theme: row.theme || 'General',
    universal_truth: row.universal_truth || '',
    url: `https://www.thedailylesson.com/lesson/${row.day_of_year}`,
  }))

  const response = {
    name: "The Daily Lesson™ Curriculum",
    version: "2026",
    total_lessons: 365,
    provider: {
      name: "Lesson of the Day, PBC",
      url: "https://www.thedailylesson.com",
      type: "Public Benefit Corporation"
    },
    license: {
      name: "CC BY-NC-SA 4.0",
      url: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      attribution: "Lesson of the Day, PBC (2026). The Daily Lesson™ Curriculum."
    },
    ages: {
      supported: ["kid", "adult", "elder"],
      range: "2-102"
    },
    languages: {
      supported: ["en", "es", "fr", "de", "pt", "zh"],
      total: 47,
      note: "47+ languages available"
    },
    api: {
      topics: "https://www.thedailylesson.com/api/curriculum/topics",
      lesson: "https://www.thedailylesson.com/api/lesson/{day}",
      today: "https://www.thedailylesson.com/api/lesson/today"
    },
    topics: topics,
    _generated: new Date().toISOString(),
  }

  return NextResponse.json(response, {
    headers: {
      'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET',
    }
  })
}
