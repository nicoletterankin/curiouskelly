import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

export async function GET() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  // Get all videos with wrong URLs
  const videos = await sql`
    SELECT video_id, age_group, archetype 
    FROM kelly_lesson_assets 
    WHERE day_number = 18 
      AND video_id IS NOT NULL 
      AND video_url NOT LIKE '%blob.vercel%'
  `

  const results = []

  for (const video of videos) {
    try {
      // Call HeyGen API to get real video status/URL
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
        {
          headers: { 'X-Api-Key': HEYGEN_API_KEY }
        }
      )

      const data = await response.json()
      
      results.push({
        video_id: video.video_id,
        age_group: video.age_group,
        archetype: video.archetype,
        status: data.data?.status || 'unknown',
        video_url: data.data?.video_url || null,
        error: data.error || null
      })
    } catch (error) {
      results.push({
        video_id: video.video_id,
        age_group: video.age_group,
        archetype: video.archetype,
        status: 'error',
        video_url: null,
        error: String(error)
      })
    }
  }

  return NextResponse.json({
    total: videos.length,
    results
  })
}

export async function POST() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  // Get all videos with wrong URLs
  const videos = await sql`
    SELECT video_id, age_group, archetype 
    FROM kelly_lesson_assets 
    WHERE day_number = 18 
      AND video_id IS NOT NULL 
      AND video_url NOT LIKE '%blob.vercel%'
  `

  const updated = []
  const failed = []

  for (const video of videos) {
    try {
      // Call HeyGen API
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
        {
          headers: { 'X-Api-Key': HEYGEN_API_KEY }
        }
      )

      const data = await response.json()

      if (data.data?.status === 'completed' && data.data?.video_url) {
        // Update database with real URL
        await sql`
          UPDATE kelly_lesson_assets 
          SET video_url = ${data.data.video_url},
              status = 'completed',
              updated_at = NOW()
          WHERE video_id = ${video.video_id}
        `

        updated.push({
          video_id: video.video_id,
          age_group: video.age_group,
          archetype: video.archetype,
          video_url: data.data.video_url
        })
      } else {
        failed.push({
          video_id: video.video_id,
          reason: data.data?.status || data.error || 'No video_url'
        })
      }
    } catch (error) {
      failed.push({
        video_id: video.video_id,
        reason: String(error)
      })
    }
  }

  return NextResponse.json({
    updated: updated.length,
    failed: failed.length,
    updatedVideos: updated,
    failedVideos: failed
  })
}
