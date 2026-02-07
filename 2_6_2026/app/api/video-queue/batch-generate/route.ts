import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * MASTER VIDEO QUEUE BATCH GENERATOR
 * 
 * Works with the video_queue table (197,100 records):
 * - 365 days × 5 phases × 3 age groups × 6 languages × 6 archetypes
 * 
 * Features:
 * - Priority-based processing (English > Spanish > others)
 * - Rate limiting and retry logic
 * - Webhook callback for completion
 * - Idempotent - safe to call multiple times
 */

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!
const HEYGEN_WEBHOOK_URL = process.env.NEXT_PUBLIC_APP_URL 
  ? `${process.env.NEXT_PUBLIC_APP_URL}/api/webhooks/heygen`
  : 'https://thedailylesson.com/api/webhooks/heygen'

// Avatar configurations by age group
const VOICES: Record<string, string> = {
  kid: '3WLAUtAsXlPNi4dw1tDM',
  adult: 'BbuMXx40WT4ZuAgRXvNx',
  senior: 'RLUgsoI4JTB52m0yY4yM',
}

// Storyteller photo IDs by age
const PHOTO_IDS: Record<string, string> = {
  kid: '1024bc304a1146998bc4c360173b2c48',
  adult: '3d6a9d6f91b444469dae87ebb3d9eba6',
  senior: '98178c87897e4421884b535b7864ba86',
}

// ============================================================================
// HEYGEN API FUNCTIONS
// ============================================================================

async function checkCredits(): Promise<{ remaining: number; used: number }> {
  try {
    const res = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY },
    })
    const data = await res.json()
    return {
      remaining: data?.data?.remaining_quota || 0,
      used: data?.data?.used_quota || 0,
    }
  } catch {
    return { remaining: 0, used: 0 }
  }
}

async function generateVideo(params: {
  id: number
  script: string
  voiceId: string
  photoId: string
  title: string
}): Promise<{ videoId: string | null; error: string | null }> {
  try {
    const res = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_inputs: [{
          character: { 
            type: 'talking_photo', 
            talking_photo_id: params.photoId 
          },
          voice: { 
            type: 'text', 
            input_text: params.script, 
            voice_id: params.voiceId 
          },
        }],
        dimension: { width: 1080, height: 1920 }, // Portrait for mobile
        title: params.title,
        callback_id: `vq_${params.id}`, // Link back to our queue ID
        callback_url: HEYGEN_WEBHOOK_URL,
      }),
    })
    
    const data = await res.json()
    if (data?.data?.video_id) {
      return { videoId: data.data.video_id, error: null }
    }
    return { videoId: null, error: data?.error?.message || JSON.stringify(data) }
  } catch (err) {
    return { videoId: null, error: String(err) }
  }
}

// ============================================================================
// GET: Status and Statistics
// ============================================================================

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const detailed = searchParams.get('detailed') === 'true'
  
  try {
    // Overall statistics
    const stats = await sql`
      SELECT 
        status,
        COUNT(*)::int as count,
        COUNT(DISTINCT day_number)::int as unique_days,
        MIN(created_at) as oldest,
        MAX(updated_at) as newest
      FROM video_queue
      GROUP BY status
      ORDER BY count DESC
    `
    
    // By language breakdown
    const byLanguage = await sql`
      SELECT 
        language,
        status,
        COUNT(*)::int as count
      FROM video_queue
      GROUP BY language, status
      ORDER BY language, status
    `
    
    // By age group breakdown
    const byAge = await sql`
      SELECT 
        age_group,
        status,
        COUNT(*)::int as count
      FROM video_queue
      GROUP BY age_group, status
      ORDER BY age_group, status
    `
    
    // Credits check
    const credits = await checkCredits()
    
    // Calculate totals
    const total = stats.reduce((sum, s) => sum + Number(s.count), 0)
    const completed = stats.find(s => s.status === 'completed')?.count || 0
    const processing = stats.find(s => s.status === 'processing')?.count || 0
    const pending = stats.find(s => s.status === 'pending')?.count || 0
    const failed = stats.find(s => s.status === 'failed')?.count || 0
    
    const response: Record<string, unknown> = {
      queue: {
        total,
        completed: Number(completed),
        processing: Number(processing),
        pending: Number(pending),
        failed: Number(failed),
        percentComplete: total > 0 ? Math.round((Number(completed) / total) * 100 * 100) / 100 : 0,
      },
      credits,
      estimatedTimeRemaining: {
        atCurrentRate: `${Math.ceil(Number(pending) / 60)} hours`, // ~60 videos/hour
        description: 'Assuming 1 video per minute with HeyGen rate limits',
      },
    }
    
    if (detailed) {
      response.byLanguage = byLanguage
      response.byAge = byAge
      response.statusDetails = stats
    }
    
    return NextResponse.json(response)
  } catch (error) {
    console.error('[VideoQueue] Status error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// ============================================================================
// POST: Process Batch
// ============================================================================

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}))
    const {
      batchSize = 10,
      dryRun = false,
      language = null,     // Filter by language
      ageGroup = null,     // Filter by age group  
      dayNumber = null,    // Filter by specific day
      phase = null,        // Filter by phase
      priorityMin = 0,     // Minimum priority
    } = body
    
    // Check credits
    const { remaining: credits } = await checkCredits()
    if (!dryRun && credits < batchSize) {
      return NextResponse.json({ 
        error: `Insufficient credits: ${credits} remaining, need ${batchSize}` 
      }, { status: 400 })
    }
    
    // Build dynamic query for pending videos
    let query = sql`
      SELECT 
        id, day_number, phase, age_group, language, archetype,
        script_text, priority, attempts
      FROM video_queue
      WHERE status = 'pending'
        AND attempts < max_attempts
        AND priority >= ${priorityMin}
    `
    
    // Note: For dynamic filtering, we'll use a single comprehensive query
    const pendingVideos = await sql`
      SELECT 
        id, day_number, phase, age_group, language, archetype,
        script_text, priority, attempts
      FROM video_queue
      WHERE status = 'pending'
        AND attempts < max_attempts
        AND priority >= ${priorityMin}
        AND (${language}::text IS NULL OR language = ${language})
        AND (${ageGroup}::text IS NULL OR age_group = ${ageGroup})
        AND (${dayNumber}::int IS NULL OR day_number = ${dayNumber})
        AND (${phase}::text IS NULL OR phase = ${phase})
      ORDER BY priority DESC, day_number ASC, 
        CASE phase 
          WHEN 'hook' THEN 1 
          WHEN 'story' THEN 2 
          WHEN 'wonder' THEN 3 
          WHEN 'action' THEN 4 
          WHEN 'wisdom' THEN 5 
        END
      LIMIT ${batchSize}
    `
    
    if (pendingVideos.length === 0) {
      return NextResponse.json({ 
        message: 'No pending videos matching criteria',
        filters: { language, ageGroup, dayNumber, phase, priorityMin }
      })
    }
    
    const results: {
      generated: Array<{ id: number; videoId: string; title: string }>
      skipped: Array<{ id: number; reason: string }>
      failed: Array<{ id: number; error: string }>
    } = { generated: [], skipped: [], failed: [] }
    
    for (const video of pendingVideos) {
      const title = `Day${video.day_number}-${video.phase}-${video.age_group}-${video.language}`
      
      // Validate script
      if (!video.script_text || video.script_text.trim().length < 10) {
        results.skipped.push({ id: video.id, reason: 'Script too short or missing' })
        continue
      }
      
      // Get voice and photo IDs
      const voiceId = VOICES[video.age_group]
      const photoId = PHOTO_IDS[video.age_group]
      
      if (!voiceId || !photoId) {
        results.skipped.push({ id: video.id, reason: `No avatar config for age: ${video.age_group}` })
        continue
      }
      
      if (dryRun) {
        results.generated.push({ 
          id: video.id, 
          videoId: 'DRY_RUN', 
          title 
        })
        continue
      }
      
      // Mark as processing
      await sql`
        UPDATE video_queue 
        SET status = 'processing', 
            started_at = NOW(),
            updated_at = NOW(),
            attempts = attempts + 1
        WHERE id = ${video.id}
      `
      
      // Generate video
      const { videoId, error } = await generateVideo({
        id: video.id,
        script: video.script_text,
        voiceId,
        photoId,
        title,
      })
      
      if (videoId) {
        await sql`
          UPDATE video_queue 
          SET heygen_video_id = ${videoId},
              updated_at = NOW()
          WHERE id = ${video.id}
        `
        results.generated.push({ id: video.id, videoId, title })
      } else {
        await sql`
          UPDATE video_queue 
          SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
              error_message = ${error},
              last_error_at = NOW(),
              updated_at = NOW()
          WHERE id = ${video.id}
        `
        results.failed.push({ id: video.id, error: error || 'Unknown error' })
      }
      
      // Rate limiting: 1.5 second delay between API calls
      await new Promise(r => setTimeout(r, 1500))
    }
    
    return NextResponse.json({
      status: dryRun ? 'dry_run' : 'batch_started',
      batchSize: pendingVideos.length,
      filters: { language, ageGroup, dayNumber, phase, priorityMin },
      summary: {
        generated: results.generated.length,
        skipped: results.skipped.length,
        failed: results.failed.length,
      },
      creditsUsed: results.generated.length,
      creditsRemaining: credits - results.generated.length,
      results,
    })
  } catch (error) {
    console.error('[VideoQueue] Batch error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// ============================================================================
// PATCH: Retry Failed Videos
// ============================================================================

export async function PATCH(request: NextRequest) {
  try {
    const { resetFailed = false, resetProcessingOlderThan = null } = await request.json().catch(() => ({}))
    
    let updated = 0
    
    if (resetFailed) {
      // Reset failed videos back to pending
      const result = await sql`
        UPDATE video_queue 
        SET status = 'pending',
            attempts = 0,
            error_message = NULL,
            updated_at = NOW()
        WHERE status = 'failed'
        RETURNING id
      `
      updated += result.length
    }
    
    if (resetProcessingOlderThan) {
      // Reset stale processing videos (stuck jobs)
      const minutes = parseInt(resetProcessingOlderThan, 10) || 30
      const result = await sql`
        UPDATE video_queue 
        SET status = 'pending',
            updated_at = NOW()
        WHERE status = 'processing'
          AND started_at < NOW() - INTERVAL '${sql.unsafe(minutes.toString())} minutes'
        RETURNING id
      `
      updated += result.length
    }
    
    return NextResponse.json({
      message: 'Reset complete',
      updated,
    })
  } catch (error) {
    console.error('[VideoQueue] Reset error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
