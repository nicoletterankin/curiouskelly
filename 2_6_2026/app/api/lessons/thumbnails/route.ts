import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/lessons/thumbnails?start=1&end=31
 * Returns lesson thumbnails for calendar display
 * 
 * Sources (in priority order):
 * 1. kelly_lesson_assets.main_visual_url (has day_number)
 * 2. teaser_images.image_url (has day column)
 * 3. Generated visual path fallback
 */
export async function GET(req: NextRequest) {
  try {
    const startDay = parseInt(req.nextUrl.searchParams.get('start') || '1')
    const endDay = parseInt(req.nextUrl.searchParams.get('end') || '365')

    // Query kelly_lesson_assets for thumbnails (primary source)
    // Use visual_url column (the actual column name in the table)
    const assets = await sql`
      SELECT DISTINCT ON (kla.day_number)
        kla.day_number as day_of_year,
        l.title,
        kla.visual_url as thumbnail_url
      FROM kelly_lesson_assets kla
      LEFT JOIN lessons l ON l.day_of_year = kla.day_number
      WHERE kla.day_number >= ${startDay} 
        AND kla.day_number <= ${endDay}
        AND kla.visual_url IS NOT NULL
      ORDER BY kla.day_number, kla.phase
    `

    // If we didn't get enough, try teaser_images as fallback
    const foundDays = new Set(assets.map((a: { day_of_year: number }) => a.day_of_year))
    
    const missingDays: number[] = []
    for (let d = startDay; d <= endDay; d++) {
      if (!foundDays.has(d)) missingDays.push(d)
    }

    let teasers: Array<{ day: number; title: string; image_url: string }> = []
    if (missingDays.length > 0) {
      teasers = await sql`
        SELECT 
          t.day as day_of_year,
          l.title,
          t.image_url as thumbnail_url
        FROM teaser_images t
        LEFT JOIN lessons l ON l.day_of_year = t.day
        WHERE t.day = ANY(${missingDays})
          AND t.image_url IS NOT NULL
          AND t.status = 'completed'
      `
    }

    // Merge DB results
    const dbThumbnails = [
      ...assets.map((a: { day_of_year: number; title: string; thumbnail_url: string }) => ({
        dayOfYear: a.day_of_year,
        title: a.title || "Today's Lesson",
        thumbnailUrl: a.thumbnail_url,
      })),
      ...teasers.map((t: { day: number; title: string; image_url: string }) => ({
        dayOfYear: t.day,
        title: t.title || "Today's Lesson",
        thumbnailUrl: t.image_url,
      })),
    ]

    // Build complete list with static file fallbacks for days 1-90
    const thumbnailsMap = new Map(dbThumbnails.map(t => [t.dayOfYear, t]))
    const thumbnails: Array<{ dayOfYear: number; title: string; thumbnailUrl: string }> = []
    
    for (let day = startDay; day <= endDay; day++) {
      if (thumbnailsMap.has(day)) {
        thumbnails.push(thumbnailsMap.get(day)!)
      } else if (day <= 90) {
        // Static fallback for days 1-90 (we have /visuals/hook/day-XXX.jpg)
        thumbnails.push({
          dayOfYear: day,
          title: "Today's Lesson",
          thumbnailUrl: `/visuals/hook/day-${day.toString().padStart(3, '0')}.jpg`,
        })
      }
    }

    // Sort by day
    thumbnails.sort((a, b) => a.dayOfYear - b.dayOfYear)

    return NextResponse.json({ thumbnails })
  } catch (error) {
    console.error('[lessons/thumbnails] Error:', error)
    return NextResponse.json({ thumbnails: [] })
  }
}
