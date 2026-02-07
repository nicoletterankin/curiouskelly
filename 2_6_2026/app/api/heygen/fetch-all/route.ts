import { getSql } from '@/lib/db'
import { NextResponse } from 'next/server'

// HeyGen API to list all videos in your account
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

interface HeyGenVideo {
  video_id: string
  status: string
  video_url?: string
  thumbnail_url?: string
  duration?: number
  created_at?: number
}

export async function GET() {
  try {
    // Fetch all videos from HeyGen (they have a list endpoint)
    const response = await fetch('https://api.heygen.com/v1/video.list?limit=100', {
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
      },
    })

    if (!response.ok) {
      const error = await response.text()
      return NextResponse.json({ error: 'HeyGen API error', details: error }, { status: 500 })
    }

    const data = await response.json()
    const videos: HeyGenVideo[] = data.data?.videos || []

    const sql = getSql()
    // Get existing video IDs from database to avoid duplicates
    const existing = await sql`SELECT heygen_video_id FROM heygen_videos WHERE heygen_video_id IS NOT NULL`
    const existingIds = new Set(existing.map((r: any) => r.heygen_video_id))

    // Filter to only completed videos not already in DB
    const newVideos = videos.filter(v => 
      v.status === 'completed' && 
      v.video_url && 
      !existingIds.has(v.video_id)
    )

    return NextResponse.json({
      total_in_heygen: videos.length,
      completed: videos.filter(v => v.status === 'completed').length,
      already_saved: existingIds.size,
      new_to_save: newVideos.length,
      videos: newVideos.map(v => ({
        video_id: v.video_id,
        video_url: v.video_url,
        thumbnail_url: v.thumbnail_url,
        duration: v.duration,
      }))
    })
  } catch (error) {
    console.error('[v0] HeyGen fetch error:', error)
    return NextResponse.json({ error: 'Failed to fetch from HeyGen', details: String(error) }, { status: 500 })
  }
}

// POST: Save all fetched videos to database
// You'll need to provide metadata mapping (which video is which age/archetype)
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const sql = getSql()
    
    // If specific video mappings provided
    if (body.mappings && Array.isArray(body.mappings)) {
      const results = []
      
      for (const mapping of body.mappings) {
        const { video_id, video_url, day, phase, age, archetype } = mapping
        
        // Save to heygen_videos
        await sql`
          INSERT INTO heygen_videos (
            day_of_year, phase, age_category, archetype,
            heygen_video_id, video_url, status, created_at
          ) VALUES (
            ${day}, ${phase}, ${age}, ${archetype},
            ${video_id}, ${video_url}, 'completed', NOW()
          )
          ON CONFLICT (day_of_year, phase, age_category, archetype) 
          DO UPDATE SET video_url = ${video_url}, heygen_video_id = ${video_id}, status = 'completed'
        `
        
        // Also save to kelly_lesson_assets for the app to display
        await sql`
          INSERT INTO kelly_lesson_assets (
            id, day_number, phase, age_group, archetype, language,
            video_url, video_id, video_source, status, created_at, updated_at
          ) VALUES (
            gen_random_uuid(), ${day}, ${phase}, ${age}, ${archetype}, 'en',
            ${video_url}, ${video_id}, 'heygen', 'completed', NOW(), NOW()
          )
          ON CONFLICT (day_number, phase, age_group, archetype, language)
          DO UPDATE SET video_url = ${video_url}, video_id = ${video_id}, status = 'completed', updated_at = NOW()
        `
        
        results.push({ video_id, saved: true })
      }
      
      return NextResponse.json({ success: true, saved: results.length, results })
    }
    
    // If just a single video URL provided (like from blob upload)
    if (body.video_url && body.day && body.phase && body.age && body.archetype) {
      const { video_url, video_id, day, phase, age, archetype } = body
      
      await sql`
        INSERT INTO heygen_videos (
          day_of_year, phase, age_category, archetype,
          heygen_video_id, video_url, status, created_at
        ) VALUES (
          ${day}, ${phase}, ${age}, ${archetype},
          ${video_id || 'manual-upload'}, ${video_url}, 'completed', NOW()
        )
        ON CONFLICT (day_of_year, phase, age_category, archetype) 
        DO UPDATE SET video_url = ${video_url}, status = 'completed'
      `
      
      await sql`
        INSERT INTO kelly_lesson_assets (
          id, day_number, phase, age_group, archetype, language,
          video_url, video_id, video_source, status, created_at, updated_at
        ) VALUES (
          gen_random_uuid(), ${day}, ${phase}, ${age}, ${archetype}, 'en',
          ${video_url}, ${video_id || 'manual-upload'}, 'heygen', 'completed', NOW(), NOW()
        )
        ON CONFLICT (day_number, phase, age_group, archetype, language)
        DO UPDATE SET video_url = ${video_url}, status = 'completed', updated_at = NOW()
      `
      
      return NextResponse.json({ success: true, message: 'Video saved' })
    }
    
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
  } catch (error) {
    console.error('[v0] Save error:', error)
    return NextResponse.json({ error: 'Failed to save', details: String(error) }, { status: 500 })
  }
}
