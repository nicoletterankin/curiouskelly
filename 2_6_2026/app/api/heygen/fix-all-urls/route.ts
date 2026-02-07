import { NextResponse } from "next/server"
import { getSql } from "@/lib/db"

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

export async function POST() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: "HEYGEN_API_KEY not configured" }, { status: 500 })
  }

  const sql = getSql()
  const results: { video_id: string; status: string; url?: string; error?: string }[] = []

  try {
    // Get all videos with wrong URLs
    const videos = await sql`
      SELECT video_id, day_number, phase, age_group, archetype
      FROM kelly_lesson_assets
      WHERE video_url LIKE '%resource.heygen.ai%'
        AND video_id IS NOT NULL
    `

    console.log(`[v0] Found ${videos.length} videos to fix`)

    for (const video of videos) {
      try {
        // Fetch real URL from HeyGen API
        const res = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
          { headers: { "X-Api-Key": HEYGEN_API_KEY } }
        )

        if (!res.ok) {
          results.push({ video_id: video.video_id, status: "api_error", error: `HTTP ${res.status}` })
          continue
        }

        const data = await res.json()
        const realUrl = data.data?.video_url

        if (!realUrl) {
          results.push({ video_id: video.video_id, status: "no_url", error: "No video_url in response" })
          continue
        }

        // Update database with real URL
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${realUrl}, updated_at = NOW()
          WHERE video_id = ${video.video_id}
        `

        results.push({ video_id: video.video_id, status: "success", url: realUrl })
        console.log(`[v0] Fixed ${video.video_id}: ${realUrl}`)

      } catch (err) {
        results.push({ 
          video_id: video.video_id, 
          status: "error", 
          error: err instanceof Error ? err.message : "Unknown" 
        })
      }
    }

    return NextResponse.json({
      total: videos.length,
      success: results.filter(r => r.status === "success").length,
      failed: results.filter(r => r.status !== "success").length,
      results
    })

  } catch (error) {
    return NextResponse.json({ 
      error: "Failed", 
      details: error instanceof Error ? error.message : "Unknown" 
    }, { status: 500 })
  }
}

export async function GET() {
  const sql = getSql()
  
  // Note: kelly_lesson_assets has audio_url, not video_url
  // Check heygen_videos table instead for video URLs
  const pending = await sql`
    SELECT COUNT(*)::int as count FROM heygen_videos WHERE video_url LIKE '%resource.heygen.ai%'
  `
  const fixed = await sql`
    SELECT COUNT(*)::int as count FROM heygen_videos WHERE video_url NOT LIKE '%resource.heygen.ai%' AND video_url IS NOT NULL
  `

  return NextResponse.json({
    pending_fixes: pending[0]?.count || 0,
    already_fixed: fixed[0]?.count || 0,
    action: "POST to this endpoint to fix all URLs"
  })
}
