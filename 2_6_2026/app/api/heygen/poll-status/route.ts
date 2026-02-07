import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

// Map age categories to age_group values used in kelly_lesson_assets
const AGE_TO_GROUP: Record<string, string> = {
  kid: 'kid',
  adult: 'adult', 
  senior: 'senior'
}

async function checkHeygenStatus(videoId: string): Promise<{
  status: string
  videoUrl?: string
  duration?: number
  error?: string
}> {
  try {
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! }
    })
    
    const data = await res.json()
    
    if (data.data?.status === 'completed') {
      return {
        status: 'completed',
        videoUrl: data.data.video_url,
        duration: data.data.duration
      }
    } else if (data.data?.status === 'failed') {
      return {
        status: 'failed',
        error: data.data.error || 'Video generation failed'
      }
    } else {
      return { status: data.data?.status || 'unknown' }
    }
  } catch (e) {
    return { status: 'error', error: e instanceof Error ? e.message : 'Unknown error' }
  }
}

export const maxDuration = 300

export async function POST(request: NextRequest) {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }
  
  const body = await request.json().catch(() => ({}))
  const limit = Math.min(body.limit || 100, 200) // Increased for mega batch
  
  // Get all processing/pending videos
  const processingVideos = await sql`
    SELECT id, day_of_year, phase, age_category, archetype, language, heygen_video_id
    FROM heygen_videos
    WHERE status IN ('processing', 'pending')
      AND heygen_video_id IS NOT NULL
    ORDER BY created_at ASC
    LIMIT ${limit}
  `
  
  if (processingVideos.length === 0) {
    return NextResponse.json({
      message: 'No videos currently processing',
      checked: 0,
      completed: 0,
      failed: 0
    })
  }
  
  const results: Array<{
    id: string
    day: number
    phase: string
    age: string
    archetype: string
    status: string
    videoUrl?: string
    error?: string
  }> = []
  
  let completedCount = 0
  let failedCount = 0
  
  for (const video of processingVideos) {
    const status = await checkHeygenStatus(video.heygen_video_id)
    
    if (status.status === 'completed' && status.videoUrl) {
      // Update heygen_videos table
      await sql`
        UPDATE heygen_videos 
        SET status = 'completed',
            video_url = ${status.videoUrl},
            duration_seconds = ${status.duration || null},
            completed_at = NOW(),
            updated_at = NOW()
        WHERE id = ${video.id}
      `
      
      // Also update kelly_lesson_assets for v0 to query
      const videoLang = video.language || 'en'
      await sql`
        INSERT INTO kelly_lesson_assets (
          id, day_number, phase, age_group, archetype, language,
          video_url, status, created_at
        ) VALUES (
          gen_random_uuid(),
          ${video.day_of_year},
          ${video.phase},
          ${AGE_TO_GROUP[video.age_category] || video.age_category},
          ${video.archetype},
          ${videoLang},
          ${status.videoUrl},
          'completed',
          NOW()
        )
        ON CONFLICT (day_number, phase, age_group, archetype, language)
        DO UPDATE SET
          video_url = ${status.videoUrl},
          status = 'completed'
      `.catch(e => console.error('[poll-status] kelly_lesson_assets upsert failed:', e))
      
      completedCount++
      results.push({
        id: video.id,
        day: video.day_of_year,
        phase: video.phase,
        age: video.age_category,
        archetype: video.archetype,
        status: 'completed',
        videoUrl: status.videoUrl
      })
      
    } else if (status.status === 'failed') {
      await sql`
        UPDATE heygen_videos 
        SET status = 'failed',
            error_message = ${status.error || 'Unknown error'},
            updated_at = NOW()
        WHERE id = ${video.id}
      `
      
      failedCount++
      results.push({
        id: video.id,
        day: video.day_of_year,
        phase: video.phase,
        age: video.age_category,
        archetype: video.archetype,
        status: 'failed',
        error: status.error
      })
      
    } else {
      // Still processing
      results.push({
        id: video.id,
        day: video.day_of_year,
        phase: video.phase,
        age: video.age_category,
        archetype: video.archetype,
        status: status.status
      })
    }
    
    // Small delay between API calls
    await new Promise(resolve => setTimeout(resolve, 200))
  }
  
  return NextResponse.json({
    message: `Checked ${processingVideos.length} videos`,
    checked: processingVideos.length,
    completed: completedCount,
    failed: failedCount,
    stillProcessing: processingVideos.length - completedCount - failedCount,
    results
  })
}

export async function GET() {
  // Get summary of all video statuses
  const summary = await sql`
    SELECT 
      status,
      COUNT(*) as count,
      COUNT(DISTINCT day_of_year) as days
    FROM heygen_videos
    GROUP BY status
  `
  
  const processingCount = await sql`
    SELECT COUNT(*) as count 
    FROM heygen_videos 
    WHERE status = 'processing'
  `
  
  const recentCompleted = await sql`
    SELECT day_of_year, phase, age_category, archetype, video_url, completed_at
    FROM heygen_videos
    WHERE status = 'completed'
    ORDER BY completed_at DESC
    LIMIT 10
  `
  
  return NextResponse.json({
    summary: summary.reduce((acc, row) => {
      acc[row.status] = { count: parseInt(row.count), days: parseInt(row.days) }
      return acc
    }, {} as Record<string, { count: number; days: number }>),
    processingCount: parseInt(processingCount[0]?.count || '0'),
    recentCompleted: recentCompleted.map(v => ({
      key: `day${v.day_of_year}-${v.phase}-${v.age_category}-${v.archetype}`,
      videoUrl: v.video_url,
      completedAt: v.completed_at
    })),
    pollEndpoint: 'POST /api/heygen/poll-status with { limit: 20 }'
  })
}
