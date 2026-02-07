import { NextResponse } from "next/server"
import { getSql } from "@/lib/db"
import { put } from "@vercel/blob"

export async function POST() {
  const sql = getSql()
  const results: { video_id: string; status: string; error?: string; blob_url?: string }[] = []

  try {
    // Get all videos with temp HeyGen URLs (files2.heygen.ai or resource.heygen.ai)
    const videos = await sql`
      SELECT video_id, day_number, phase, age_group, archetype, video_url
      FROM kelly_lesson_assets
      WHERE (video_url LIKE '%heygen.ai%' OR video_url LIKE '%files2.heygen%')
        AND video_url NOT LIKE '%blob.vercel%'
      ORDER BY day_number, phase
      LIMIT 20
    `

    console.log(`[v0] Found ${videos.length} videos to sync`)

    for (const video of videos) {
      const videoId = video.video_id || `day${video.day_number}-${video.phase}-${video.age_group}`
      
      try {
        // Use existing video_url directly (already have the temp HeyGen URL)
        const heygenVideoUrl = video.video_url
        
        if (!heygenVideoUrl) {
          results.push({ video_id: videoId, status: "error", error: "No video_url found" })
          continue
        }

        console.log(`[v0] Using existing HeyGen URL for ${videoId}`)

        // Step 2: Download video from HeyGen
        console.log(`[v0] Downloading video ${videoId}`)
        const videoRes = await fetch(heygenVideoUrl)
        
        if (!videoRes.ok) {
          results.push({ video_id: videoId, status: "error", error: `Failed to download: ${videoRes.status}` })
          continue
        }

        const videoBlob = await videoRes.blob()
        console.log(`[v0] Downloaded ${(videoBlob.size / 1024 / 1024).toFixed(2)}MB`)

        // Step 3: Upload to Vercel Blob
        const blobPath = `kelly-videos/day-${video.day_number}/${video.phase}/${video.age_group}-${video.archetype}-${videoId}.mp4`
        console.log(`[v0] Uploading to Blob: ${blobPath}`)
        
        const blob = await put(blobPath, videoBlob, {
          access: "public",
          contentType: "video/mp4",
        })

        console.log(`[v0] Uploaded to: ${blob.url}`)

        // Step 4: Update database with Blob URL
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${blob.url}, updated_at = NOW()
          WHERE day_number = ${video.day_number}
            AND phase = ${video.phase}
            AND age_group = ${video.age_group}
        `

        results.push({ video_id: videoId, status: "success", blob_url: blob.url })
        console.log(`[v0] Successfully synced ${videoId}`)

      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Unknown error"
        results.push({ video_id: videoId, status: "error", error: errorMsg })
        console.error(`[v0] Error syncing ${videoId}:`, err)
      }
    }

    const successCount = results.filter(r => r.status === "success").length
    const errorCount = results.filter(r => r.status === "error").length

    return NextResponse.json({
      total: videos.length,
      success: successCount,
      errors: errorCount,
      results,
    })

  } catch (error) {
    console.error("[v0] Bulk sync error:", error)
    return NextResponse.json({ 
      error: "Bulk sync failed", 
      details: error instanceof Error ? error.message : "Unknown error" 
    }, { status: 500 })
  }
}

export async function GET() {
  const sql = getSql()
  
  try {
    const pendingCount = await sql`
      SELECT COUNT(*)::int as count
      FROM kelly_lesson_assets
      WHERE (video_url LIKE '%heygen.ai%' OR video_url LIKE '%files2.heygen%')
        AND video_url NOT LIKE '%blob.vercel%'
    `
    
    const completedCount = await sql`
      SELECT COUNT(*)::int as count
      FROM kelly_lesson_assets
      WHERE video_url LIKE '%blob.vercel-storage%'
    `

    return NextResponse.json({
      pending: pendingCount[0]?.count || 0,
      completed: completedCount[0]?.count || 0,
      message: "POST to this endpoint to sync pending videos to Vercel Blob"
    })
  } catch (error) {
    return NextResponse.json({ error: "Failed to check status" }, { status: 500 })
  }
}
