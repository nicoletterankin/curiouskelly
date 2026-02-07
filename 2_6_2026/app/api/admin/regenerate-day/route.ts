/**
 * ADMIN: Regenerate a specific day's audio and video assets
 * 
 * POST /api/admin/regenerate-day
 * Body: { day: 34, step: 'audio' | 'video' | 'status' }
 */

import { NextRequest, NextResponse } from 'next/server'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom']
const AGE_GROUPS = ['kid', 'adult', 'senior']
const ARCHETYPES = ['explorer', 'scientist', 'artist', 'athlete', 'provider', 'empath', 'mystic', 'macgyver']

export async function POST(request: NextRequest) {
  try {
    const { day, step = 'status' } = await request.json()
    
    if (!day || day < 1 || day > 365) {
      return NextResponse.json({ error: 'Invalid day (1-365)' }, { status: 400 })
    }

    // Step 1: Check current status
    const sql = getSql()
    const assets = await sql`
      SELECT 
        phase, age_group, archetype,
        audio_url IS NOT NULL as has_audio,
        video_url IS NOT NULL as has_video,
        status
      FROM kelly_lesson_assets 
      WHERE day_number = ${day}
      ORDER BY phase, age_group
    `

    const lesson = await sql`
      SELECT day_of_year, title, hook_script FROM lessons WHERE day_of_year = ${day}
    `

    const summary = {
      day,
      lesson: lesson[0] || null,
      totalAssets: assets.length,
      withAudio: assets.filter((a: Record<string, unknown>) => a.has_audio).length,
      withVideo: assets.filter((a: Record<string, unknown>) => a.has_video).length,
      pending: assets.filter((a: Record<string, unknown>) => a.status === 'pending').length,
    }

    if (step === 'status') {
      return NextResponse.json({ step: 'status', ...summary })
    }

    // Step 2: Generate Audio
    if (step === 'audio') {
      const needsAudio = assets.filter((a: Record<string, unknown>) => !a.has_audio)
      
      if (needsAudio.length === 0) {
        return NextResponse.json({ 
          step: 'audio', 
          message: 'All assets already have audio',
          ...summary 
        })
      }

      // Call audio/batch endpoint
      const baseUrl = request.nextUrl.origin
      const audioResponse = await fetch(`${baseUrl}/api/audio/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          days: [day],
          phases: PHASES,
          ageGroup: 'adult',
          language: 'en',
          dryRun: false,
        }),
      })

      const audioResult = await audioResponse.json()
      
      return NextResponse.json({
        step: 'audio',
        message: `Queued audio generation for day ${day}`,
        audioResult,
        ...summary,
      })
    }

    // Step 3: Generate Videos
    if (step === 'video') {
      const needsVideo = assets.filter((a: Record<string, unknown>) => a.has_audio && !a.has_video)
      
      if (needsVideo.length === 0) {
        const noAudio = assets.filter((a: Record<string, unknown>) => !a.has_audio).length
        return NextResponse.json({ 
          step: 'video', 
          message: noAudio > 0 
            ? `${noAudio} assets need audio first. Run step=audio first.`
            : 'All assets already have video',
          ...summary 
        })
      }

      // Queue video generation for first batch (to avoid overwhelming HeyGen)
      const batch = needsVideo.slice(0, 5)
      const baseUrl = request.nextUrl.origin
      const results = []

      for (const asset of batch) {
        try {
          const videoResponse = await fetch(`${baseUrl}/api/heygen/pipeline`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              day_number: day,
              phase: asset.phase,
              age_group: asset.age_group,
              archetype: asset.archetype,
            }),
          })
          const result = await videoResponse.json()
          results.push({ ...asset, result })
        } catch (e) {
          results.push({ ...asset, error: String(e) })
        }
      }

      return NextResponse.json({
        step: 'video',
        message: `Queued ${batch.length} videos for day ${day}. ${needsVideo.length - batch.length} remaining.`,
        results,
        remaining: needsVideo.length - batch.length,
        ...summary,
      })
    }

    return NextResponse.json({ error: 'Invalid step. Use: status, audio, or video' }, { status: 400 })

  } catch (error) {
    console.error('[admin/regenerate-day] Error:', error)
    return NextResponse.json({ 
      error: 'Regeneration failed', 
      details: String(error) 
    }, { status: 500 })
  }
}

// GET for quick status check
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '34')

  const sql = getSql()
  const assets = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets 
    WHERE day_number = ${day}
  `

  const lesson = await sql`
    SELECT day_of_year, title FROM lessons WHERE day_of_year = ${day}
  `

  return NextResponse.json({
    day,
    lesson: lesson[0] || null,
    assets: assets[0],
  })
}
