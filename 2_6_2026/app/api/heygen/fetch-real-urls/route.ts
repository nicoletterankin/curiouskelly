import { getSql } from "@/lib/db"
import { NextResponse } from "next/server"

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Fetches real video URLs from HeyGen API and updates database
export async function POST() {
  try {
    const sql = getSql()
    // Get all videos with video_id but wrong/missing URLs
    const videos = await sql`
      SELECT id, video_id, video_url 
      FROM kelly_lesson_assets 
      WHERE video_id IS NOT NULL 
      AND (video_url LIKE '%resource.heygen.ai%' OR video_url IS NULL)
    `

    const results = []

    for (const video of videos) {
      try {
        // Fetch real URL from HeyGen API
        const res = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
          {
            headers: { "X-Api-Key": HEYGEN_API_KEY },
          }
        )

        const data = await res.json()
        
        if (data.data?.video_url) {
          // Update database with real URL
          await sql`
            UPDATE kelly_lesson_assets 
            SET video_url = ${data.data.video_url}, 
                updated_at = NOW()
            WHERE id = ${video.id}
          `
          results.push({
            video_id: video.video_id,
            status: "updated",
            new_url: data.data.video_url,
          })
        } else {
          results.push({
            video_id: video.video_id,
            status: "no_url",
            heygen_status: data.data?.status || "unknown",
          })
        }
      } catch (err) {
        results.push({
          video_id: video.video_id,
          status: "error",
          error: String(err),
        })
      }
    }

    return NextResponse.json({
      total: videos.length,
      results,
    })
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// GET lists all HeyGen videos from their API
export async function GET() {
  try {
    const res = await fetch("https://api.heygen.com/v1/video.list?limit=100", {
      headers: { "X-Api-Key": HEYGEN_API_KEY },
    })

    const data = await res.json()
    
    // Get details for each completed video
    const videos = data.data?.videos || []
    const completed = videos.filter((v: { status: string }) => v.status === "completed")

    const withUrls = await Promise.all(
      completed.slice(0, 30).map(async (v: { video_id: string }) => {
        const detailRes = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${v.video_id}`,
          { headers: { "X-Api-Key": HEYGEN_API_KEY } }
        )
        const detail = await detailRes.json()
        return {
          video_id: v.video_id,
          video_url: detail.data?.video_url,
          thumbnail_url: detail.data?.thumbnail_url,
          duration: detail.data?.duration,
        }
      })
    )

    return NextResponse.json({
      total: completed.length,
      videos: withUrls,
    })
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
