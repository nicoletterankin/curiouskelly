import { NextResponse } from 'next/server'
import { generateLipSyncVideo, getBaseVideo, BASE_VIDEOS } from '@/lib/fal-lipsync-client'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'

// Map all 8 age groups to the 3 base video ages
function mapAgeToBaseAge(ageGroup: string): 'kid' | 'adult' | 'senior' {
  switch (ageGroup) {
    case 'toddler':
    case 'kid':
      return 'kid'
    case 'preteen':
    case 'youngAdult':
    case 'adult':
    case 'middleAge':
      return 'adult'
    case 'senior':
    case 'elder':
      return 'senior'
    default:
      return 'adult'
  }
}

interface Asset {
  id: number
  day_number: number
  phase: string
  age_group: string
  archetype: string
  audio_url: string
}

/**
 * Process a single asset - generates lipsync video and uploads to Blob
 */
async function processAsset(asset: Asset): Promise<{
  success: boolean
  id: number
  videoUrl?: string
  error?: string
}> {
  try {
    const baseAge = mapAgeToBaseAge(asset.age_group)
    const baseVideoUrl = getBaseVideo(baseAge, asset.archetype)
    
    if (!baseVideoUrl) {
      throw new Error(`No base video for ${baseAge}/${asset.archetype}`)
    }
    
    // Generate lip-synced video
    const result = await generateLipSyncVideo({
      videoUrl: baseVideoUrl,
      audioUrl: asset.audio_url,
      syncMode: 'cut',
    })
    
    // Upload to Vercel Blob
    const videoResponse = await fetch(result.videoUrl)
    const videoBuffer = await videoResponse.arrayBuffer()
    
    const blobPath = `video/kelly/day-${String(asset.day_number).padStart(3, '0')}/${asset.phase}-${asset.age_group}-${asset.archetype}.mp4`
    const blob = await put(blobPath, videoBuffer, {
      access: 'public',
      contentType: 'video/mp4',
      addRandomSuffix: false,
    })
    
    // Update database
    const sql = getSql()
    await sql`
      UPDATE kelly_lesson_assets 
      SET video_url = ${blob.url},
          video_source = 'fal_lipsync',
          status = 'completed',
          updated_at = NOW()
      WHERE id = ${asset.id}
    `
    
    return { success: true, id: asset.id, videoUrl: blob.url }
  } catch (error) {
    console.error(`[mass-lipsync] Error processing ${asset.id}:`, error)
    return { 
      success: false, 
      id: asset.id, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }
  }
}

/**
 * POST /api/fal/mass-lipsync
 * 
 * Mass parallel lip-sync generation for ALL ready assets
 * 
 * Body: { 
 *   concurrency?: number (default 10),
 *   limit?: number (default 100),
 *   dayStart?: number,
 *   dayEnd?: number,
 *   ageGroup?: string
 * }
 */
export async function POST(request: Request) {
  const startTime = Date.now()
  
  try {
    const body = await request.json()
    const { 
      concurrency = 10, 
      limit = 100,
      dayStart,
      dayEnd,
      ageGroup 
    } = body
    
    console.log(`[mass-lipsync] Starting with concurrency=${concurrency}, limit=${limit}`)
    
    const sql = getSql()
    // Build dynamic query based on filters
    let assets: Asset[]
    
    if (dayStart && dayEnd && ageGroup) {
      assets = await sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE audio_url IS NOT NULL
          AND video_url IS NULL
          AND day_number >= ${dayStart}
          AND day_number <= ${dayEnd}
          AND age_group = ${ageGroup}
        ORDER BY day_number, phase
        LIMIT ${limit}
      ` as Asset[]
    } else if (dayStart && dayEnd) {
      assets = await sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE audio_url IS NOT NULL
          AND video_url IS NULL
          AND day_number >= ${dayStart}
          AND day_number <= ${dayEnd}
        ORDER BY day_number, phase
        LIMIT ${limit}
      ` as Asset[]
    } else if (ageGroup) {
      assets = await sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE audio_url IS NOT NULL
          AND video_url IS NULL
          AND age_group = ${ageGroup}
        ORDER BY day_number, phase
        LIMIT ${limit}
      ` as Asset[]
    } else {
      assets = await sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE audio_url IS NOT NULL
          AND video_url IS NULL
        ORDER BY day_number, phase
        LIMIT ${limit}
      ` as Asset[]
    }
    
    if (assets.length === 0) {
      return NextResponse.json({
        success: true,
        message: 'No assets to process',
        processed: 0,
      })
    }
    
    console.log(`[mass-lipsync] Found ${assets.length} assets to process`)
    
    // Process in parallel batches
    const results: { success: boolean; id: number; videoUrl?: string; error?: string }[] = []
    
    for (let i = 0; i < assets.length; i += concurrency) {
      const batch = assets.slice(i, i + concurrency)
      console.log(`[mass-lipsync] Processing batch ${Math.floor(i / concurrency) + 1}/${Math.ceil(assets.length / concurrency)} (${batch.length} items)`)
      
      const batchResults = await Promise.all(batch.map(processAsset))
      results.push(...batchResults)
      
      const succeeded = batchResults.filter(r => r.success).length
      console.log(`[mass-lipsync] Batch complete: ${succeeded}/${batch.length} succeeded`)
    }
    
    const succeeded = results.filter(r => r.success).length
    const failed = results.filter(r => !r.success).length
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
    
    // Get remaining count
    const remaining = await sql`
      SELECT COUNT(*) as count
      FROM kelly_lesson_assets 
      WHERE audio_url IS NOT NULL
        AND video_url IS NULL
    `
    
    return NextResponse.json({
      success: true,
      summary: {
        processed: results.length,
        succeeded,
        failed,
        remaining: parseInt(remaining[0].count as string),
        elapsedSeconds: parseFloat(elapsed),
        videosPerSecond: (succeeded / parseFloat(elapsed)).toFixed(2),
      },
      failures: results.filter(r => !r.success).slice(0, 10),
    })
    
  } catch (error) {
    console.error('[mass-lipsync] Error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

/**
 * GET /api/fal/mass-lipsync
 * 
 * Get status of ready assets
 */
export async function GET() {
  const sql = getSql()
  // Count ready assets by age group
  const byAge = await sql`
    SELECT 
      age_group,
      COUNT(*) as ready
    FROM kelly_lesson_assets
    WHERE audio_url IS NOT NULL
      AND video_url IS NULL
    GROUP BY age_group
    ORDER BY ready DESC
  `
  
  // Count ready assets by day (first 30)
  const byDay = await sql`
    SELECT 
      day_number,
      COUNT(*) as ready
    FROM kelly_lesson_assets
    WHERE audio_url IS NOT NULL
      AND video_url IS NULL
    GROUP BY day_number
    ORDER BY day_number
    LIMIT 30
  `
  
  // Total counts
  const totals = await sql`
    SELECT 
      COUNT(*) as total_slots,
      COUNT(audio_url) as has_audio,
      COUNT(video_url) as has_video,
      COUNT(CASE WHEN audio_url IS NOT NULL AND video_url IS NULL THEN 1 END) as ready_for_lipsync
    FROM kelly_lesson_assets
  `
  
  // Base video coverage
  const baseVideoCount = Object.keys(BASE_VIDEOS).length
  
  return NextResponse.json({
    success: true,
    totals: totals[0],
    byAgeGroup: byAge,
    byDay: byDay,
    baseVideosAvailable: baseVideoCount,
    instructions: {
      startSmall: 'POST { "limit": 10, "concurrency": 5 }',
      mediumBatch: 'POST { "limit": 100, "concurrency": 10 }',
      largeBatch: 'POST { "limit": 500, "concurrency": 15 }',
      filterByDay: 'POST { "dayStart": 1, "dayEnd": 30, "limit": 500 }',
      filterByAge: 'POST { "ageGroup": "senior", "limit": 200 }',
    },
  })
}
