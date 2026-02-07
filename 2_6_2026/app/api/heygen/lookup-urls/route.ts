import { NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY || process.env.HEYGEN_KEY

export async function GET() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  const sql = getSql()
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
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
        { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
      )
      const data = await response.json()

      results.push({
        video_id: video.video_id,
        age_group: video.age_group,
        archetype: video.archetype,
        status: data.data?.status,
        video_url: data.data?.video_url,
        error: data.error?.message
      })
    } catch (err) {
      results.push({
        video_id: video.video_id,
        error: String(err)
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

  const sql = getSql()
  // Get all videos with wrong URLs
  const videos = await sql`
    SELECT video_id, age_group, archetype 
    FROM kelly_lesson_assets 
    WHERE day_number = 18 
      AND video_id IS NOT NULL 
      AND video_url NOT LIKE '%blob.vercel%'
  `

  const results = []
  let updated = 0
  let failed = 0

  for (const video of videos) {
    try {
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
        { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
      )
      const data = await response.json()

      if (data.data?.status === 'completed' && data.data?.video_url) {
        // Update database with real URL
        await sql`
          UPDATE kelly_lesson_assets 
          SET video_url = ${data.data.video_url}, updated_at = NOW()
          WHERE video_id = ${video.video_id}
        `
        updated++
        results.push({
          video_id: video.video_id,
          age_group: video.age_group,
          archetype: video.archetype,
          status: 'updated',
          new_url: data.data.video_url
        })
      } else {
        failed++
        results.push({
          video_id: video.video_id,
          age_group: video.age_group,
          archetype: video.archetype,
          status: data.data?.status || 'unknown',
          error: data.error?.message || 'Video not ready'
        })
      }
    } catch (err) {
      failed++
      results.push({
        video_id: video.video_id,
        error: String(err)
      })
    }
  }

  return NextResponse.json({ 
    total: videos.length,
    updated,
    failed,
    results 
  })
}
