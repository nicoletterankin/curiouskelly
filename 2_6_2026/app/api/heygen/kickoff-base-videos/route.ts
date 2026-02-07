import { NextRequest, NextResponse } from 'next/server'
import { getAllAvatarConfigs, heygenClient } from '@/lib/heygen-client'
import { getSql } from '@/lib/db'

// Base script for all avatars (5-10 seconds)
const BASE_SCRIPT = `Hello! I'm Kelly, and I'm here to share something wonderful with you today. Let's discover something amazing together!`

interface VideoJob {
  avatarName: string
  ageCategory: string
  archetype: string
  avatarGroupId: string
  lookId: string
  videoId?: string
  status: 'pending' | 'generating' | 'completed' | 'failed'
  videoUrl?: string
  error?: string
}

/**
 * POST /api/heygen/kickoff-base-videos
 * 
 * Starts generation of all 36 base avatar videos.
 * Returns immediately with job IDs - poll /api/heygen/video-status/[id] for results.
 * 
 * Query params:
 * - test=true: Use test mode (faster, lower quality)
 * - limit=N: Only generate first N videos (for testing)
 */
export async function POST(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const testMode = searchParams.get('test') === 'true'
    const limit = searchParams.get('limit') ? parseInt(searchParams.get('limit')!) : undefined

    // Get all 36 avatar configurations
    let configs = getAllAvatarConfigs()
    
    if (limit && limit > 0) {
      configs = configs.slice(0, limit)
    }

    console.log(`[HeyGen Kickoff] Starting generation of ${configs.length} base videos (test=${testMode})`)

    // Check credits first
    let creditsInfo = { remaining: 0, used: 0 }
    try {
      creditsInfo = await heygenClient.getCredits()
      console.log(`[HeyGen Kickoff] Credits: ${creditsInfo.remaining} remaining, ${creditsInfo.used} used`)
      
      if (creditsInfo.remaining < configs.length) {
        return NextResponse.json({
          success: false,
          error: `Not enough credits. Need ${configs.length}, have ${creditsInfo.remaining}`,
          credits: creditsInfo,
        }, { status: 400 })
      }
    } catch (e) {
      console.warn('[HeyGen Kickoff] Could not check credits:', e)
    }

    // Initialize jobs
    const jobs: VideoJob[] = configs.map(config => ({
      avatarName: config.avatarName,
      ageCategory: config.ageCategory,
      archetype: config.archetype,
      avatarGroupId: config.avatarGroupId,
      lookId: config.lookId,
      status: 'pending',
    }))

    // Start all video generations
    const results = await Promise.allSettled(
      jobs.map(async (job, index) => {
        // Stagger requests to avoid rate limiting (200ms between each)
        await new Promise(r => setTimeout(r, index * 200))
        
        try {
          console.log(`[HeyGen Kickoff] Starting ${job.avatarName}...`)
          job.status = 'generating'
          
          const result = await heygenClient.generateVideo({
            avatarGroupId: job.avatarGroupId,
            lookId: job.lookId,
            script: BASE_SCRIPT,
            title: `Base Video - ${job.avatarName}`,
            test: testMode,
          })
          
          job.videoId = result.video_id
          job.status = 'generating'
          console.log(`[HeyGen Kickoff] ${job.avatarName} started: ${result.video_id}`)
          
          // Save to database
          const sql = getSql()
          await sql`
            INSERT INTO heygen_videos (
              heygen_video_id, video_type, age_category, archetype, script, status
            ) VALUES (
              ${result.video_id}, 'base', ${job.ageCategory}, ${job.archetype}, ${BASE_SCRIPT}, 'processing'
            )
            ON CONFLICT (heygen_video_id) DO UPDATE SET
              status = 'processing',
              updated_at = NOW()
          `
          
          return job
        } catch (error) {
          job.status = 'failed'
          job.error = error instanceof Error ? error.message : 'Unknown error'
          console.error(`[HeyGen Kickoff] ${job.avatarName} FAILED:`, job.error)
          throw error
        }
      })
    )

    // Summarize results
    const successful = jobs.filter(j => j.videoId)
    const failed = jobs.filter(j => j.status === 'failed')

    return NextResponse.json({
      success: true,
      summary: {
        total: jobs.length,
        started: successful.length,
        failed: failed.length,
        testMode,
      },
      credits: creditsInfo,
      jobs: jobs.map(j => ({
        avatarName: j.avatarName,
        videoId: j.videoId,
        status: j.status,
        error: j.error,
      })),
      // Convenience: Poll these IDs for status
      videoIds: successful.map(j => j.videoId).filter(Boolean),
      // Next step instructions
      nextSteps: {
        pollStatus: `/api/heygen/video-status/[videoId]`,
        checkAll: `/api/heygen/check-base-videos?ids=${successful.map(j => j.videoId).join(',')}`,
      },
    })
  } catch (error) {
    console.error('[HeyGen Kickoff] Error:', error)
    
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Kickoff failed',
    }, { status: 500 })
  }
}

/**
 * GET /api/heygen/kickoff-base-videos
 * 
 * Returns info about what will be generated.
 */
export async function GET() {
  const configs = getAllAvatarConfigs()
  
  let credits = { remaining: 0, used: 0 }
  try {
    credits = await heygenClient.getCredits()
  } catch {
    // Ignore
  }
  
  return NextResponse.json({
    avatarCount: configs.length,
    credits,
    canGenerate: credits.remaining >= configs.length,
    avatars: configs.map(c => ({
      name: c.avatarName,
      age: c.ageCategory,
      archetype: c.archetype,
    })),
    baseScript: BASE_SCRIPT,
    estimatedCost: `~${configs.length} credits (~$${(configs.length * 0.1).toFixed(2)})`,
    usage: {
      start: 'POST /api/heygen/kickoff-base-videos',
      testMode: 'POST /api/heygen/kickoff-base-videos?test=true',
      limitedTest: 'POST /api/heygen/kickoff-base-videos?test=true&limit=3',
    },
  })
}
