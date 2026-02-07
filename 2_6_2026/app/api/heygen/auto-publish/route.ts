import { NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Auto-publish: Poll HeyGen for completed videos, download, upload to Blob, update DB
export async function POST() {
  try {
    const sql = getSql()
    // Find videos that are marked as 'processing' or have HeyGen URLs (not Blob)
    const pending = await sql`
      SELECT id, video_id, day_number, phase, age_group, archetype, video_url
      FROM kelly_lesson_assets
      WHERE video_id IS NOT NULL
        AND (
          status = 'processing'
          OR video_url LIKE '%heygen%'
        )
      LIMIT 10
    `

    if (pending.length === 0) {
      return NextResponse.json({ message: 'No videos to process', processed: 0 })
    }

    const results = []

    for (const video of pending) {
      try {
        // Get video status from HeyGen
        const statusRes = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
          {
            headers: { 'X-Api-Key': HEYGEN_API_KEY },
          }
        )
        const statusData = await statusRes.json()

        if (statusData.data?.status !== 'completed') {
          results.push({
            video_id: video.video_id,
            status: statusData.data?.status || 'unknown',
            action: 'skipped - not completed',
          })
          continue
        }

        const heygenUrl = statusData.data.video_url
        if (!heygenUrl) {
          results.push({
            video_id: video.video_id,
            status: 'completed',
            action: 'skipped - no URL',
          })
          continue
        }

        // Download video from HeyGen
        const videoRes = await fetch(heygenUrl)
        if (!videoRes.ok) {
          results.push({
            video_id: video.video_id,
            action: 'failed - download error',
          })
          continue
        }

        const videoBlob = await videoRes.blob()

        // Upload to Vercel Blob
        const filename = `kelly/day${video.day_number}/${video.phase}/${video.age_group}-${video.archetype}.mp4`
        const blob = await put(filename, videoBlob, {
          access: 'public',
          contentType: 'video/mp4',
        })

        // Update database with Blob URL
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${blob.url},
              status = 'completed',
              updated_at = NOW()
          WHERE id = ${video.id}
        `

        results.push({
          video_id: video.video_id,
          day: video.day_number,
          phase: video.phase,
          age_group: video.age_group,
          archetype: video.archetype,
          action: 'published',
          blob_url: blob.url,
        })
      } catch (error) {
        results.push({
          video_id: video.video_id,
          action: 'error',
          error: error instanceof Error ? error.message : 'Unknown error',
        })
      }
    }

    return NextResponse.json({
      processed: results.length,
      published: results.filter((r) => r.action === 'published').length,
      results,
    })
  } catch (error) {
    console.error('Auto-publish error:', error)
    return NextResponse.json(
      { error: 'Auto-publish failed' },
      { status: 500 }
    )
  }
}

// GET: Check status of videos pending publish
export async function GET() {
  try {
    const sql = getSql()
    const pending = await sql`
      SELECT day_number, phase, age_group, archetype, status, 
             CASE WHEN video_url LIKE '%blob.vercel%' THEN 'blob' ELSE 'heygen' END as url_type
      FROM kelly_lesson_assets
      WHERE video_id IS NOT NULL
      ORDER BY day_number, phase, age_group, archetype
    `

    const blobCount = pending.filter((v) => v.url_type === 'blob').length
    const heygenCount = pending.filter((v) => v.url_type === 'heygen').length

    return NextResponse.json({
      total: pending.length,
      published_to_blob: blobCount,
      needs_publish: heygenCount,
      videos: pending,
    })
  } catch (error) {
    console.error('Status check error:', error)
    return NextResponse.json({ error: 'Status check failed' }, { status: 500 })
  }
}
