import { NextResponse } from "next/server"
import { sql } from '@/lib/db'

// POST /api/video/register
// Register videos that Desktop Claude uploads to R2
// FIXED: Use heygen_videos table (has video_url), not kelly_lesson_assets (audio only)
export async function POST(request: Request) {
  try {
    const { videos } = await request.json()
    
    if (!Array.isArray(videos) || videos.length === 0) {
      return NextResponse.json({ error: "videos array required" }, { status: 400 })
    }

    const results = []
    
    for (const video of videos) {
      const { 
        dayNumber, 
        phase, 
        ageGroup,
        age_category, // heygen_videos uses age_category
        language = 'en',
        archetype = 'storyteller',
        videoUrl,
        filename 
      } = video

      if (!dayNumber || !phase || !videoUrl) {
        results.push({ filename, status: 'skipped', reason: 'missing required fields' })
        continue
      }

      const ageCategory = age_category || ageGroup || 'adult'

      // Upsert into heygen_videos (the correct table for video URLs)
      await sql`
        INSERT INTO heygen_videos (day_of_year, phase, age_category, archetype, video_url, status, created_at, updated_at)
        VALUES (${dayNumber}, ${phase}, ${ageCategory}, ${archetype}, ${videoUrl}, 'completed', NOW(), NOW())
        ON CONFLICT (day_of_year, phase, age_category, archetype) 
        DO UPDATE SET video_url = ${videoUrl}, status = 'completed', updated_at = NOW()
      `
      
      results.push({ filename, dayNumber, phase, ageCategory, status: 'registered' })
    }

    return NextResponse.json({ 
      success: true, 
      registered: results.filter(r => r.status === 'registered').length,
      skipped: results.filter(r => r.status === 'skipped').length,
      results 
    })

  } catch (error) {
    console.error('[video/register] Error:', error)
    return NextResponse.json({ 
      error: "Registration failed", 
      details: (error as Error).message 
    }, { status: 500 })
  }
}

// GET /api/video/register - Check registration status
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const dayNumber = searchParams.get('day')

  try {
    if (dayNumber) {
      const videos = await sql`
        SELECT day_of_year, phase, age_category, archetype, video_url, status
        FROM heygen_videos
        WHERE day_of_year = ${parseInt(dayNumber)} AND video_url IS NOT NULL
        ORDER BY phase, age_category
      `
      return NextResponse.json({ dayNumber, videos })
    }

    // Return summary
    const summary = await sql`
      SELECT 
        COUNT(*) as total_videos,
        COUNT(DISTINCT day_of_year) as days_covered,
        COUNT(DISTINCT phase) as phases_covered,
        COUNT(DISTINCT age_category) as age_groups_covered
      FROM heygen_videos
      WHERE video_url IS NOT NULL
    `

    return NextResponse.json({ summary: summary[0] })
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 })
  }
}
