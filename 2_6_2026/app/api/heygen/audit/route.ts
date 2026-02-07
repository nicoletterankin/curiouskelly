import { NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY || process.env.HEYGEN_KEY

interface HeyGenStatusResponse {
  code: number
  data: {
    video_id: string
    status: 'pending' | 'processing' | 'completed' | 'failed'
    video_url?: string
    thumbnail_url?: string
    duration?: number
    error?: string
  }
  message?: string
}

export async function GET() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  const sql = getSql()
  // Get all video_ids from Day 18
  const videos = await sql`
    SELECT video_id, age_group, archetype, video_url 
    FROM kelly_lesson_assets 
    WHERE day_number = 18 AND video_id IS NOT NULL
    ORDER BY age_group, archetype
  `

  const results = []

  for (const video of videos) {
    const videoId = video.video_id as string
    
    // Skip the one that already has Vercel Blob URL
    if ((video.video_url as string)?.includes('blob.vercel')) {
      results.push({
        video_id: videoId,
        age_group: video.age_group,
        archetype: video.archetype,
        heygen_status: 'SKIPPED - Already has Blob URL',
        current_url: video.video_url,
        real_url: video.video_url,
        action: 'NONE'
      })
      continue
    }

    try {
      // Call HeyGen API to get real status
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
        {
          method: 'GET',
          headers: {
            'X-Api-Key': HEYGEN_API_KEY,
          },
        }
      )

      const text = await response.text()
      let data: HeyGenStatusResponse

      try {
        data = JSON.parse(text)
      } catch {
        results.push({
          video_id: videoId,
          age_group: video.age_group,
          archetype: video.archetype,
          heygen_status: 'API_ERROR',
          error: `Invalid JSON: ${text.substring(0, 200)}`,
          current_url: video.video_url,
          real_url: null,
          action: 'INVESTIGATE'
        })
        continue
      }

      if (data.code === 100 && data.data) {
        const realUrl = data.data.video_url || null
        const status = data.data.status

        results.push({
          video_id: videoId,
          age_group: video.age_group,
          archetype: video.archetype,
          heygen_status: status,
          heygen_code: data.code,
          current_url: video.video_url,
          real_url: realUrl,
          thumbnail_url: data.data.thumbnail_url,
          duration: data.data.duration,
          urls_match: video.video_url === realUrl,
          action: status === 'completed' && realUrl ? 'UPDATE_DB' : 'WAIT_OR_REGENERATE'
        })
      } else {
        results.push({
          video_id: videoId,
          age_group: video.age_group,
          archetype: video.archetype,
          heygen_status: 'API_ERROR',
          heygen_code: data.code,
          heygen_message: data.message,
          current_url: video.video_url,
          real_url: null,
          action: 'VIDEO_NOT_FOUND_OR_EXPIRED'
        })
      }
    } catch (err) {
      results.push({
        video_id: videoId,
        age_group: video.age_group,
        archetype: video.archetype,
        heygen_status: 'FETCH_ERROR',
        error: err instanceof Error ? err.message : 'Unknown error',
        current_url: video.video_url,
        real_url: null,
        action: 'RETRY'
      })
    }
  }

  // Summary
  const summary = {
    total: results.length,
    completed_with_url: results.filter(r => r.heygen_status === 'completed' && r.real_url).length,
    pending: results.filter(r => r.heygen_status === 'pending').length,
    processing: results.filter(r => r.heygen_status === 'processing').length,
    failed: results.filter(r => r.heygen_status === 'failed').length,
    not_found: results.filter(r => r.action === 'VIDEO_NOT_FOUND_OR_EXPIRED').length,
    already_blob: results.filter(r => r.action === 'NONE').length,
    errors: results.filter(r => r.heygen_status === 'API_ERROR' || r.heygen_status === 'FETCH_ERROR').length,
  }

  return NextResponse.json({
    audit_time: new Date().toISOString(),
    api_key_present: !!HEYGEN_API_KEY,
    api_key_prefix: HEYGEN_API_KEY?.substring(0, 10) + '...',
    summary,
    results
  }, { status: 200 })
}

// POST: Update database with real URLs from completed videos
export async function POST() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  const sql = getSql()
  const videos = await sql`
    SELECT video_id, age_group, archetype, video_url 
    FROM kelly_lesson_assets 
    WHERE day_number = 18 
      AND video_id IS NOT NULL
      AND video_url NOT LIKE '%blob.vercel%'
    ORDER BY age_group, archetype
  `

  const updates = []

  for (const video of videos) {
    const videoId = video.video_id as string

    try {
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
        {
          method: 'GET',
          headers: { 'X-Api-Key': HEYGEN_API_KEY },
        }
      )

      const data: HeyGenStatusResponse = await response.json()

      if (data.code === 100 && data.data?.status === 'completed' && data.data.video_url) {
        // Update database with real URL
        await sql`
          UPDATE kelly_lesson_assets 
          SET video_url = ${data.data.video_url}
          WHERE video_id = ${videoId}
        `

        updates.push({
          video_id: videoId,
          age_group: video.age_group,
          archetype: video.archetype,
          old_url: video.video_url,
          new_url: data.data.video_url,
          status: 'UPDATED'
        })
      } else {
        updates.push({
          video_id: videoId,
          age_group: video.age_group,
          archetype: video.archetype,
          status: 'SKIPPED',
          reason: data.data?.status || data.message || 'No video URL'
        })
      }
    } catch (err) {
      updates.push({
        video_id: videoId,
        status: 'ERROR',
        error: err instanceof Error ? err.message : 'Unknown'
      })
    }
  }

  return NextResponse.json({
    updated_count: updates.filter(u => u.status === 'UPDATED').length,
    skipped_count: updates.filter(u => u.status === 'SKIPPED').length,
    error_count: updates.filter(u => u.status === 'ERROR').length,
    updates
  })
}
