import { NextResponse } from "next/server"
import { getSql } from "@/lib/db"

// This endpoint fetches REAL video URLs from HeyGen API and updates our database
export async function POST() {
  const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
  
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: "HEYGEN_API_KEY not configured" }, { status: 500 })
  }

  try {
    const sql = getSql()
    // Get all video_ids from our database
    const dbVideos = await sql`
      SELECT video_id, day_number, phase, age_group, archetype 
      FROM kelly_lesson_assets 
      WHERE video_id IS NOT NULL
    `

    const results = []

    for (const dbVideo of dbVideos) {
      try {
        // Fetch real URL from HeyGen
        const response = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${dbVideo.video_id}`,
          { headers: { "X-Api-Key": HEYGEN_API_KEY } }
        )
        
        const data = await response.json()
        
        if (data.data?.status === "completed" && data.data?.video_url) {
          // Update with real URL
          await sql`
            UPDATE kelly_lesson_assets 
            SET video_url = ${data.data.video_url}, updated_at = NOW()
            WHERE video_id = ${dbVideo.video_id}
          `
          
          results.push({
            video_id: dbVideo.video_id,
            age: dbVideo.age_group,
            archetype: dbVideo.archetype,
            status: "updated",
            real_url: data.data.video_url
          })
        } else {
          results.push({
            video_id: dbVideo.video_id,
            status: data.data?.status || "unknown",
            error: "Not completed or no URL"
          })
        }
      } catch (err) {
        results.push({
          video_id: dbVideo.video_id,
          status: "error",
          error: String(err)
        })
      }
    }

    return NextResponse.json({
      processed: results.length,
      results
    })
  } catch (error) {
    console.error("Error fixing URLs:", error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

export async function GET() {
  const sql = getSql()
  // Show current state
  const videos = await sql`
    SELECT day_number, phase, age_group, archetype, video_url, video_id
    FROM kelly_lesson_assets 
    WHERE video_id IS NOT NULL
    ORDER BY age_group, archetype
  `
  
  return NextResponse.json({
    count: videos.length,
    videos
  })
}
