import { NextResponse } from 'next/server'
import { generateLessonVideo, getBaseVideo, BASE_VIDEOS } from '@/lib/fal-lipsync-client'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'
import { put } from '@vercel/blob'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

/**
 * POST /api/fal/batch-day
 * 
 * Generate ALL videos for a specific day using fal.ai lip-sync
 * Reuses Day 18 HeyGen base videos + new audio
 * 
 * Body: { dayNumber, ageGroup?, archetype?, limit? }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { dayNumber, ageGroup, archetype, limit = 5 } = body
    
    if (!dayNumber) {
      return NextResponse.json({
        success: false,
        error: 'Missing required field: dayNumber'
      }, { status: 400 })
    }
    
    console.log(`[batch-day] Starting batch generation for Day ${dayNumber}`)
    
    // 1. Get all audio files for this day that need video generation
    const sql = getSql()
    let query = sql`
      SELECT id, day_number, phase, age_group, archetype, audio_url
      FROM kelly_lesson_assets 
      WHERE day_number = ${dayNumber}
        AND audio_url IS NOT NULL
        AND (video_url IS NULL OR video_source != 'fal_lipsync')
    `
    
    // Optional filters
    if (ageGroup) {
      query = sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE day_number = ${dayNumber}
          AND audio_url IS NOT NULL
          AND age_group = ${ageGroup}
          AND (video_url IS NULL OR video_source != 'fal_lipsync')
      `
    }
    
    const assets = await query
    
    if (assets.length === 0) {
      return NextResponse.json({
        success: false,
        error: `No audio assets found for Day ${dayNumber} that need video generation`,
        hint: 'Either all videos are already generated, or audio is missing'
      }, { status: 404 })
    }
    
    console.log(`[batch-day] Found ${assets.length} assets to process, limiting to ${limit}`)
    
    // Filter to ones we have base videos for
    const processable = assets.filter((a: any) => {
      const age = a.age_group === 'kid' ? 'kid' : a.age_group === 'senior' ? 'senior' : 'adult'
      const baseVideo = getBaseVideo(age, a.archetype)
      return baseVideo !== undefined
    }).slice(0, limit)
    
    console.log(`[batch-day] Processing ${processable.length} assets with available base videos`)
    
    const results: any[] = []
    
    for (const asset of processable as any[]) {
      try {
        console.log(`[batch-day] Processing: ${asset.phase}/${asset.age_group}/${asset.archetype}`)
        
        const age = asset.age_group === 'kid' ? 'kid' : asset.age_group === 'senior' ? 'senior' : 'adult'
        
        // Generate lip-synced video
        const result = await generateLessonVideo({
          audioUrl: asset.audio_url,
          age,
          archetype: asset.archetype,
        })
        
        // Upload to Vercel Blob
        const videoResponse = await fetch(result.videoUrl)
        const videoBuffer = await videoResponse.arrayBuffer()
        
        const blobPath = `video/kelly/day-${String(dayNumber).padStart(3, '0')}/${asset.phase}-${asset.age_group}-${asset.archetype}.mp4`
        const blob = await put(blobPath, videoBuffer, {
          access: 'public',
          contentType: 'video/mp4',
          addRandomSuffix: false,
        })
        
        // Update database
        await sql`
          UPDATE kelly_lesson_assets 
          SET video_url = ${blob.url},
              video_source = 'fal_lipsync',
              status = 'completed',
              updated_at = NOW()
          WHERE id = ${asset.id}
        `
        
        results.push({
          success: true,
          phase: asset.phase,
          ageGroup: asset.age_group,
          archetype: asset.archetype,
          videoUrl: blob.url,
        })
        
        console.log(`[batch-day] ✅ ${asset.phase}/${asset.age_group}/${asset.archetype}`)
        
      } catch (error) {
        console.error(`[batch-day] ❌ ${asset.phase}/${asset.age_group}/${asset.archetype}:`, error)
        
        results.push({
          success: false,
          phase: asset.phase,
          ageGroup: asset.age_group,
          archetype: asset.archetype,
          error: error instanceof Error ? error.message : 'Unknown error',
        })
      }
    }
    
    const succeeded = results.filter(r => r.success).length
    const failed = results.filter(r => !r.success).length
    
    return NextResponse.json({
      success: true,
      dayNumber,
      summary: {
        total: processable.length,
        succeeded,
        failed,
        remaining: assets.length - processable.length,
      },
      results,
    })
    
  } catch (error) {
    console.error('[batch-day] Error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

/**
 * GET /api/fal/batch-day?day=19
 * 
 * Check what assets are available for a day
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const dayNumber = parseInt(searchParams.get('day') || '19')
  
  // Get all assets for this day
  const sql = getSql()
  const assets = await sql`
    SELECT day_number, phase, age_group, archetype, 
           audio_url IS NOT NULL as has_audio,
           video_url IS NOT NULL as has_video,
           video_source,
           status
    FROM kelly_lesson_assets 
    WHERE day_number = ${dayNumber}
    ORDER BY phase, age_group, archetype
  `
  
  // Summarize
  const withAudio = assets.filter((a: any) => a.has_audio).length
  const withVideo = assets.filter((a: any) => a.has_video).length
  const needsVideo = assets.filter((a: any) => a.has_audio && !a.has_video).length
  
  // Check base video coverage
  const baseVideoCoverage = Object.keys(BASE_VIDEOS)
  
  return NextResponse.json({
    dayNumber,
    summary: {
      total: assets.length,
      withAudio,
      withVideo,
      needsVideo,
    },
    baseVideosAvailable: baseVideoCoverage,
    assets,
  })
}
