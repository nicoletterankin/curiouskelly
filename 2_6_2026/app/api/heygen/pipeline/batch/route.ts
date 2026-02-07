import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

/**
 * POST /api/heygen/pipeline/batch
 * 
 * Process ALL assets for a day that need video generation
 * Calls /api/heygen/pipeline for each one sequentially
 * 
 * Body: { day_number: 18, max_concurrent?: 1 }
 */
export async function POST(request: NextRequest) {
  const startTime = Date.now()
  
  try {
    const body = await request.json()
    const { day_number, phase_filter, max_items } = body

    if (!day_number) {
      return NextResponse.json({ error: 'day_number required' }, { status: 400 })
    }

    const sql = getSql()
    // Get all assets that need video
    let query = sql`
      SELECT day_number, phase, age_group, archetype, audio_url, video_url
      FROM kelly_lesson_assets 
      WHERE day_number = ${day_number}
        AND audio_url IS NOT NULL
        AND (video_url IS NULL OR video_url NOT LIKE '%blob.vercel%')
      ORDER BY phase, age_group, archetype
    `

    const toProcess = await query

    if (toProcess.length === 0) {
      return NextResponse.json({
        success: true,
        message: 'No assets need video generation',
        day_number
      })
    }

    // Optionally filter by phase
    let filtered = toProcess
    if (phase_filter) {
      filtered = toProcess.filter((a: Record<string, unknown>) => a.phase === phase_filter)
    }

    // Optionally limit
    if (max_items && max_items > 0) {
      filtered = filtered.slice(0, max_items)
    }

    const results: Array<{
      asset: Record<string, unknown>
      success: boolean
      video_url?: string
      error?: string
      elapsed_seconds?: number
    }> = []

    // Get the base URL for internal API calls
    const baseUrl = request.headers.get('host')
    const protocol = baseUrl?.includes('localhost') ? 'http' : 'https'

    // Process each asset
    for (const asset of filtered) {
      const assetStart = Date.now()
      console.log(`[v0 Batch] Processing: Day ${asset.day_number} ${asset.phase} ${asset.age_group}/${asset.archetype}`)

      try {
        const response = await fetch(`${protocol}://${baseUrl}/api/heygen/pipeline`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            day_number: asset.day_number,
            phase: asset.phase,
            age_group: asset.age_group,
            archetype: asset.archetype
          })
        })

        const result = await response.json()

        if (response.ok && result.success) {
          results.push({
            asset: { 
              phase: asset.phase, 
              age_group: asset.age_group, 
              archetype: asset.archetype 
            },
            success: true,
            video_url: result.video_url,
            elapsed_seconds: (Date.now() - assetStart) / 1000
          })
          console.log(`[v0 Batch] SUCCESS: ${asset.phase} ${asset.age_group}/${asset.archetype}`)
        } else {
          results.push({
            asset: { 
              phase: asset.phase, 
              age_group: asset.age_group, 
              archetype: asset.archetype 
            },
            success: false,
            error: result.error || 'Unknown error',
            elapsed_seconds: (Date.now() - assetStart) / 1000
          })
          console.log(`[v0 Batch] FAILED: ${asset.phase} ${asset.age_group}/${asset.archetype} - ${result.error}`)
        }
      } catch (e) {
        results.push({
          asset: { 
            phase: asset.phase, 
            age_group: asset.age_group, 
            archetype: asset.archetype 
          },
          success: false,
          error: e instanceof Error ? e.message : String(e),
          elapsed_seconds: (Date.now() - assetStart) / 1000
        })
        console.log(`[v0 Batch] ERROR: ${asset.phase} ${asset.age_group}/${asset.archetype} - ${e}`)
      }

      // Small delay between requests to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    const successful = results.filter(r => r.success)
    const failed = results.filter(r => !r.success)
    const totalElapsed = (Date.now() - startTime) / 1000

    return NextResponse.json({
      success: true,
      day_number,
      total_processed: results.length,
      successful: successful.length,
      failed: failed.length,
      total_elapsed_seconds: totalElapsed,
      results,
      summary: {
        success_rate: `${((successful.length / results.length) * 100).toFixed(1)}%`,
        avg_time_per_video: `${(totalElapsed / results.length).toFixed(1)}s`
      }
    })

  } catch (error) {
    console.error('[v0 Batch] Error:', error)
    return NextResponse.json({
      error: 'Batch processing failed',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 })
  }
}

/**
 * GET /api/heygen/pipeline/batch?day=18
 * 
 * Preview what would be processed
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '18')

  const sql = getSql()
  const toProcess = await sql`
    SELECT day_number, phase, age_group, archetype, 
           audio_url IS NOT NULL as has_audio,
           video_url,
           CASE 
             WHEN video_url LIKE '%blob.vercel%' THEN 'COMPLETE'
             WHEN video_url IS NOT NULL THEN 'BAD_URL'
             ELSE 'NO_VIDEO'
           END as video_status
    FROM kelly_lesson_assets 
    WHERE day_number = ${day}
    ORDER BY phase, age_group, archetype
  `

  const needsGeneration = toProcess.filter((a: Record<string, unknown>) => 
    a.has_audio && a.video_status !== 'COMPLETE'
  )

  return NextResponse.json({
    day,
    total_assets: toProcess.length,
    needs_generation: needsGeneration.length,
    already_complete: toProcess.filter((a: Record<string, unknown>) => a.video_status === 'COMPLETE').length,
    assets_to_process: needsGeneration,
    estimated_time: `${needsGeneration.length * 90} seconds (${(needsGeneration.length * 90 / 60).toFixed(1)} minutes)`,
    start_command: `POST /api/heygen/pipeline/batch with body: { "day_number": ${day} }`
  })
}
