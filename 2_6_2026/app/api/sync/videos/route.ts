import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { createClient } from '@supabase/supabase-js'
import { neon } from '@neondatabase/serverless' // Import neon here

/**
 * Video Sync API - Syncs completed videos from Supabase pipeline to Neon app database
 */

interface VideoJob {
  day_number: number
  phase: string
  age_group: string
  video_url: string
  audio_url?: string
  thumbnail_url?: string
  status: string
}

function mapAgeGroup(ageGroup: string): string {
  const mapping: Record<string, string> = {
    child: 'kid',
    kid: 'kid',
    adult: 'adult',
    elder: 'senior',
    senior: 'senior',
  }
  return mapping[ageGroup?.toLowerCase()] || 'adult'
}

export async function POST(request: NextRequest) {
  // Initialize clients inside handler to avoid build-time errors
  const sqlClient = neon(process.env.DATABASE_URL!) // Use sqlClient instead of sql to avoid conflict
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY
  
  if (!supabaseUrl || !supabaseKey) {
    return NextResponse.json({ error: 'Supabase not configured' }, { status: 500 })
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey)
  
  const { searchParams } = new URL(request.url)
  const specificDay = searchParams.get('day')
  const force = searchParams.get('force') === 'true'

  // Verify cron secret
  const authHeader = request.headers.get('authorization')
  const cronSecret = process.env.CRON_SECRET
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    // Query heygen_videos table which exists in both Supabase and Neon
    let query = supabase
      .from('heygen_videos')
      .select('*')
      .eq('status', 'completed')
      .not('video_url', 'is', null)
      .order('completed_at', { ascending: false })

    if (specificDay) {
      query = query.eq('day_number', parseInt(specificDay))
    }

    if (!force) {
      const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      query = query.gte('completed_at', oneDayAgo)
    }

    const { data: videos, error: fetchError } = await query.limit(500)

    if (fetchError) {
      return NextResponse.json({ error: 'Supabase fetch failed', details: fetchError.message }, { status: 500 })
    }

    if (!videos || videos.length === 0) {
      return NextResponse.json({ success: true, message: 'No new videos', synced: 0 })
    }

    let synced = 0
    const errors: string[] = []

    for (const video of videos as VideoJob[]) {
      try {
        const ageCategory = mapAgeGroup(video.age_group)
        await sqlClient`
          INSERT INTO heygen_videos (day_of_year, phase, age_category, video_url, audio_url, thumbnail_url, status, created_at)
          VALUES (${video.day_number}, ${video.phase}, ${ageCategory}, ${video.video_url}, ${video.audio_url || null}, ${video.thumbnail_url || null}, 'completed', NOW())
          ON CONFLICT (day_of_year, phase, age_category) 
          DO UPDATE SET video_url = EXCLUDED.video_url, audio_url = EXCLUDED.audio_url, thumbnail_url = EXCLUDED.thumbnail_url, status = 'completed', updated_at = NOW()
        `
        synced++
      } catch (err) {
        errors.push(`Day ${video.day_number} ${video.phase}: ${err instanceof Error ? err.message : 'Unknown'}`)
      }
    }

    return NextResponse.json({ success: true, synced, total: videos.length, errors: errors.slice(0, 10) })
  } catch (error) {
    return NextResponse.json({ error: 'Sync failed', details: error instanceof Error ? error.message : 'Unknown' }, { status: 500 })
  }
}

export async function GET() {
  const sqlClient = neon(process.env.DATABASE_URL!) // Use sqlClient instead of sql to avoid conflict

  try {
    const recentVideos = await sqlClient`
      SELECT day_of_year, phase, age_category, status, created_at
      FROM heygen_videos WHERE video_url IS NOT NULL
      ORDER BY created_at DESC LIMIT 10
    `
    const countResult = await sqlClient`SELECT COUNT(*) as total FROM heygen_videos WHERE video_url IS NOT NULL`

    return NextResponse.json({ status: 'ok', totalVideos: countResult[0]?.total || 0, recentVideos })
  } catch (error) {
    return NextResponse.json({ status: 'error', error: error instanceof Error ? error.message : 'Unknown' }, { status: 500 })
  }
}
