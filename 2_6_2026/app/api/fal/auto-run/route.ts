import { NextResponse } from 'next/server'
import { generateLipSyncVideo, getBaseVideo } from '@/lib/fal-lipsync-client'
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
 * GET /api/fal/auto-run
 * 
 * Automatically processes a batch of ready videos.
 * Designed to be called by Vercel Cron every minute.
 * 
 * Query params:
 * - batch: number of videos to process (default 5)
 * - prioritize: "early" (days 1-30 first) or "all" (any day)
 */
export async function GET(request: Request) {
  const startTime = Date.now()
  const { searchParams } = new URL(request.url)
  const batch = parseInt(searchParams.get('batch') || '5')
  const prioritize = searchParams.get('prioritize') || 'early'
  
  console.log(`[auto-run] Starting batch of ${batch}, prioritize=${prioritize}`)
  
  try {
    const sql = getSql()
    // Get next batch of assets to process, prioritizing early days
    let assets: Asset[]
    
    if (prioritize === 'early') {
      // First try to get from days 1-30, then fall back to any
      assets = await sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE audio_url IS NOT NULL
          AND video_url IS NULL
          AND day_number <= 30
        ORDER BY day_number, age_group, archetype
        LIMIT ${batch}
      ` as Asset[]
      
      // If none in early days, get any
      if (assets.length === 0) {
        assets = await sql`
          SELECT id, day_number, phase, age_group, archetype, audio_url
          FROM kelly_lesson_assets 
          WHERE audio_url IS NOT NULL
            AND video_url IS NULL
          ORDER BY day_number, age_group, archetype
          LIMIT ${batch}
        ` as Asset[]
      }
    } else {
      assets = await sql`
        SELECT id, day_number, phase, age_group, archetype, audio_url
        FROM kelly_lesson_assets 
        WHERE audio_url IS NOT NULL
          AND video_url IS NULL
        ORDER BY day_number, age_group, archetype
        LIMIT ${batch}
      ` as Asset[]
    }
    
    if (assets.length === 0) {
      return NextResponse.json({
        success: true,
        message: 'No assets ready for processing',
        processed: 0,
      })
    }
    
    const results: { success: boolean; id: number; day: number; error?: string }[] = []
    
    // Process all in parallel
    const promises = assets.map(async (asset) => {
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
        await sql`
          UPDATE kelly_lesson_assets 
          SET video_url = ${blob.url},
              video_source = 'fal_lipsync',
              status = 'completed',
              updated_at = NOW()
          WHERE id = ${asset.id}
        `
        
        return { success: true, id: asset.id, day: asset.day_number }
      } catch (error) {
        console.error(`[auto-run] Error processing ${asset.id}:`, error)
        return { 
          success: false, 
          id: asset.id, 
          day: asset.day_number,
          error: error instanceof Error ? error.message : 'Unknown error' 
        }
      }
    })
    
    const batchResults = await Promise.all(promises)
    results.push(...batchResults)
    
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
    
    // Get total completed
    const completed = await sql`
      SELECT COUNT(*) as count
      FROM kelly_lesson_assets 
      WHERE video_url IS NOT NULL
    `
    
    console.log(`[auto-run] Completed: ${succeeded}/${batch} in ${elapsed}s, ${remaining[0].count} remaining`)
    
    return NextResponse.json({
      success: true,
      batch: {
        requested: batch,
        processed: results.length,
        succeeded,
        failed,
      },
      progress: {
        totalCompleted: parseInt(completed[0].count as string),
        remaining: parseInt(remaining[0].count as string),
        elapsedSeconds: parseFloat(elapsed),
      },
      daysProcessed: [...new Set(results.filter(r => r.success).map(r => r.day))],
      failures: results.filter(r => !r.success),
    })
    
  } catch (error) {
    console.error('[auto-run] Error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

// Also support POST for manual triggers with custom params
export async function POST(request: Request) {
  const body = await request.json()
  const batch = body.batch || 10
  const prioritize = body.prioritize || 'early'
  
  // Reconstruct URL with params and call GET
  const url = new URL(request.url)
  url.searchParams.set('batch', batch.toString())
  url.searchParams.set('prioritize', prioritize)
  
  return GET(new Request(url.toString()))
}
