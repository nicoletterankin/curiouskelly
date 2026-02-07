import { NextResponse } from "next/server"
import { sql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

export async function POST(request: Request) {
  try {
    const { batchSize = 10, priority = "hook" } = await request.json()

    if (!HEYGEN_API_KEY) {
      return NextResponse.json({ error: "HeyGen API key not configured" }, { status: 500 })
    }

    // Get queued videos, prioritizing hook phase
    const queuedVideos = await sql`
      SELECT id, lesson_id, avatar_id, voice_id, script, video_input_url, status
      FROM heygen_videos
      WHERE status = 'queued'
      ORDER BY 
        CASE WHEN script LIKE '%hook%' THEN 0 ELSE 1 END,
        created_at ASC
      LIMIT ${batchSize}
    `

    if (queuedVideos.length === 0) {
      return NextResponse.json({ message: "No videos in queue" })
    }

    const results = []
    const errors = []

    for (const video of queuedVideos) {
      try {
        // Mark as processing
        await sql`
          UPDATE heygen_videos SET status = 'processing', updated_at = NOW()
          WHERE id = ${video.id}
        `

        // Call HeyGen API to create video
        const response = await fetch("https://api.heygen.com/v2/video/generate", {
          method: "POST",
          headers: {
            "X-Api-Key": HEYGEN_API_KEY,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            video_inputs: [
              {
                character: {
                  type: "avatar",
                  avatar_id: video.avatar_id || "Kristin_pubblic_2_20230901",
                  avatar_style: "normal",
                },
                voice: {
                  type: "text",
                  input_text: video.script,
                  voice_id: video.voice_id || "1bd001e7e50f421d891986aad5158bc8",
                },
                background: {
                  type: "color",
                  value: "#FFFFFF",
                },
              },
            ],
            dimension: {
              width: 1280,
              height: 720,
            },
            aspect_ratio: "16:9",
          }),
        })

        if (!response.ok) {
          const error = await response.text()
          await sql`
            UPDATE heygen_videos 
            SET status = 'error', error_message = ${error}, updated_at = NOW()
            WHERE id = ${video.id}
          `
          errors.push({ id: video.id, error })
          continue
        }

        const data = await response.json()
        const videoId = data.data?.video_id

        if (videoId) {
          await sql`
            UPDATE heygen_videos 
            SET heygen_video_id = ${videoId}, status = 'generating', updated_at = NOW()
            WHERE id = ${video.id}
          `
          results.push({ id: video.id, heygenVideoId: videoId })
        }
      } catch (err) {
        await sql`
          UPDATE heygen_videos 
          SET status = 'error', error_message = ${String(err)}, updated_at = NOW()
          WHERE id = ${video.id}
        `
        errors.push({ id: video.id, error: String(err) })
      }
    }

    return NextResponse.json({
      success: true,
      processed: results.length,
      errors: errors.length,
      results,
      errors: errors.length > 0 ? errors : undefined,
    })
  } catch (error) {
    console.error("HeyGen queue processor error:", error)
    return NextResponse.json(
      { error: "Failed to process queue" },
      { status: 500 }
    )
  }
}

// GET: Check queue status
export async function GET() {
  const stats = await sql`
    SELECT status, COUNT(*) as count
    FROM heygen_videos
    GROUP BY status
    ORDER BY count DESC
  `

  const total = stats.reduce((sum, s) => sum + Number(s.count), 0)
  const completed = stats.find(s => s.status === "completed")?.count || 0

  return NextResponse.json({
    total,
    byStatus: Object.fromEntries(stats.map(s => [s.status, Number(s.count)])),
    percentComplete: Math.round((Number(completed) / total) * 100),
  })
}
