import { getSql } from "@/lib/db"
import { generateLessonVideo, BASE_VIDEOS } from "@/lib/fal-lipsync-client"
import { put } from "@vercel/blob"

export const maxDuration = 300

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const batchSize = parseInt(searchParams.get("batch") || "5")
  const concurrency = parseInt(searchParams.get("concurrency") || "2")
  
  const sql = getSql()
  
  // Get assets ready for lipsync (have audio, no video)
  const assets = await sql`
    SELECT id, day_number, phase, age_group, archetype, audio_url
    FROM kelly_lesson_assets
    WHERE audio_url IS NOT NULL
      AND video_url IS NULL
      AND status = 'adapted'
    ORDER BY day_number, id
    LIMIT ${batchSize}
  `
  
  if (assets.length === 0) {
    return Response.json({ 
      success: true, 
      message: "No assets ready for lipsync",
      processed: 0 
    })
  }
  
  const results: { id: string; success: boolean; error?: string; videoUrl?: string }[] = []
  
  // Process in chunks based on concurrency
  for (let i = 0; i < assets.length; i += concurrency) {
    const chunk = assets.slice(i, i + concurrency)
    
    const chunkResults = await Promise.allSettled(
      chunk.map(async (asset) => {
        // Mark as processing
        await sql`
          UPDATE kelly_lesson_assets
          SET status = 'lipsync_processing', updated_at = NOW()
          WHERE id = ${asset.id}
        `
        
        // Check if we have a base video for this combo
        const key = `${asset.age_group}_${asset.archetype}`
        if (!BASE_VIDEOS[key]) {
          throw new Error(`No base video for ${key}`)
        }
        
        // Generate lip-synced video
        const result = await generateLessonVideo({
          audioUrl: asset.audio_url,
          age: asset.age_group as 'kid' | 'adult' | 'senior',
          archetype: asset.archetype,
        })
        
        // Upload to Vercel Blob for permanent storage
        const videoResponse = await fetch(result.videoUrl)
        const videoBlob = await videoResponse.blob()
        const blobResult = await put(
          `video/kelly/lessons/day${asset.day_number}-${asset.phase}-${asset.age_group}-${asset.archetype}.mp4`,
          videoBlob,
          { access: 'public', contentType: 'video/mp4' }
        )
        
        // Update database with completed video
        await sql`
          UPDATE kelly_lesson_assets
          SET 
            video_url = ${blobResult.url},
            video_source = 'fal_lipsync',
            status = 'completed',
            updated_at = NOW()
          WHERE id = ${asset.id}
        `
        
        return { id: asset.id, videoUrl: blobResult.url }
      })
    )
    
    for (let j = 0; j < chunkResults.length; j++) {
      const result = chunkResults[j]
      const asset = chunk[j]
      if (result.status === "fulfilled") {
        results.push({ id: asset.id, success: true, videoUrl: result.value.videoUrl })
      } else {
        // Mark as failed in DB
        await sql`
          UPDATE kelly_lesson_assets
          SET status = 'adapted', error_message = ${result.reason?.message || 'Unknown error'}, updated_at = NOW()
          WHERE id = ${asset.id}
        `
        results.push({ id: asset.id, success: false, error: result.reason?.message || "Unknown error" })
      }
    }
  }
  
  const successCount = results.filter(r => r.success).length
  const failCount = results.filter(r => !r.success).length
  
  return Response.json({
    success: true,
    message: `Completed ${successCount} videos, ${failCount} failed`,
    processed: successCount,
    failed: failCount,
    results
  })
}
