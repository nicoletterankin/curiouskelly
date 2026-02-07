import { NextRequest, NextResponse } from 'next/server'
import { getAllAvatarConfigs, heygenClient, getAvatarId } from '@/lib/heygen-client'
import { getSql } from '@/lib/db'

const BASE_SCRIPT = "Hello! I'm Kelly, your curious learning companion. Every day brings something amazing to discover together!"

// GET: Check status of all base videos
export async function GET() {
  try {
    // Check credits first
    const credits = await heygenClient.getCredits()
    
    const sql = getSql()
    // Get status from database
    const dbStatus = await sql`
      SELECT 
        COUNT(*) FILTER (WHERE status = 'completed') as completed,
        COUNT(*) FILTER (WHERE status = 'processing') as processing,
        COUNT(*) FILTER (WHERE status = 'failed') as failed,
        COUNT(*) FILTER (WHERE status = 'pending') as pending,
        COUNT(*) as total
      FROM heygen_videos 
      WHERE video_type = 'base'
    `
    
    // Get any processing videos that need status check
    const processingVideos = await sql`
      SELECT heygen_video_id, age_category, archetype, created_at
      FROM heygen_videos 
      WHERE video_type = 'base' AND status = 'processing'
      ORDER BY created_at ASC
      LIMIT 10
    `
    
    // Check status of processing videos
    const statusUpdates = []
    for (const video of processingVideos) {
      try {
        const status = await heygenClient.getVideoStatus(video.heygen_video_id)
        
        if (status.status === 'completed' && status.video_url) {
          await sql`
            UPDATE heygen_videos SET
              status = 'completed',
              video_url = ${status.video_url},
              thumbnail_url = ${status.thumbnail_url || null},
              duration_seconds = ${status.duration || null},
              completed_at = NOW(),
              updated_at = NOW()
            WHERE heygen_video_id = ${video.heygen_video_id}
          `
          statusUpdates.push({ id: video.heygen_video_id, status: 'completed', url: status.video_url })
        } else if (status.status === 'failed') {
          await sql`
            UPDATE heygen_videos SET
              status = 'failed',
              error_message = ${status.error || 'Unknown error'},
              updated_at = NOW()
            WHERE heygen_video_id = ${video.heygen_video_id}
          `
          statusUpdates.push({ id: video.heygen_video_id, status: 'failed', error: status.error })
        } else {
          statusUpdates.push({ id: video.heygen_video_id, status: status.status })
        }
      } catch (e) {
        statusUpdates.push({ id: video.heygen_video_id, status: 'error', error: String(e) })
      }
    }
    
    const stats = dbStatus[0] || { completed: 0, processing: 0, failed: 0, pending: 0, total: 0 }
    
    return NextResponse.json({
      credits: credits.remaining,
      avatarsConfigured: getAllAvatarConfigs().length,
      database: {
        completed: Number(stats.completed),
        processing: Number(stats.processing),
        failed: Number(stats.failed),
        pending: Number(stats.pending),
        total: Number(stats.total)
      },
      statusUpdates,
      nextAction: Number(stats.completed) === 36 
        ? 'All base videos complete!' 
        : Number(stats.processing) > 0 
          ? 'Videos still processing. Poll again in 30 seconds.'
          : 'POST to this endpoint to generate missing videos.'
    })
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// POST: Generate missing base videos
export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '36', 10)
    const testMode = searchParams.get('test') === 'true'
    
    // Check credits
    const credits = await heygenClient.getCredits()
    if (credits.remaining < 1) {
      return NextResponse.json({ error: 'No HeyGen credits remaining' }, { status: 400 })
    }
    
    // Get all avatar configs
    const allAvatars = getAllAvatarConfigs()
    
    const sql = getSql()
    // Get already generated videos from database
    const existingVideos = await sql`
      SELECT age_category, archetype, status
      FROM heygen_videos 
      WHERE video_type = 'base' AND status IN ('completed', 'processing')
    `
    
    const existingSet = new Set(
      existingVideos.map(v => `${v.age_category}-${v.archetype}`)
    )
    
    // Find missing avatars
    const missingAvatars = allAvatars.filter(
      a => !existingSet.has(`${a.ageCategory}-${a.archetype}`)
    )
    
    if (missingAvatars.length === 0) {
      return NextResponse.json({ 
        message: 'All base videos already generated or processing',
        existing: existingVideos.length
      })
    }
    
    // Limit how many to generate
    const toGenerate = missingAvatars.slice(0, Math.min(limit, credits.remaining))
    
    const results = []
    
    for (const avatar of toGenerate) {
      try {
        const avatarId = getAvatarId(avatar.ageCategory, avatar.archetype)
        
        if (!avatarId) {
          results.push({ 
            avatar: `${avatar.ageCategory}_${avatar.archetype}`, 
            status: 'skipped', 
            error: 'No avatar ID configured' 
          })
          continue
        }
        
        // Generate video via HeyGen API
        const response = await heygenClient.generateVideo({
          avatarId,
          script: BASE_SCRIPT,
          title: `Kelly Base - ${avatar.ageCategory} ${avatar.archetype}`,
          test: testMode
        })
        
        // Save to database
        await sql`
          INSERT INTO heygen_videos (
            heygen_video_id, video_type, age_category, archetype, script, status
          ) VALUES (
            ${response.video_id}, 'base', ${avatar.ageCategory}, ${avatar.archetype}, ${BASE_SCRIPT}, 'processing'
          )
          ON CONFLICT DO NOTHING
        `
        
        results.push({
          avatar: `${avatar.ageCategory}_${avatar.archetype}`,
          videoId: response.video_id,
          status: 'started'
        })
        
        // Small delay to avoid rate limiting
        await new Promise(r => setTimeout(r, 500))
        
      } catch (e) {
        results.push({
          avatar: `${avatar.ageCategory}_${avatar.archetype}`,
          status: 'failed',
          error: String(e)
        })
      }
    }
    
    const started = results.filter(r => r.status === 'started').length
    const failed = results.filter(r => r.status === 'failed').length
    
    return NextResponse.json({
      message: `Started ${started} video generations${failed > 0 ? `, ${failed} failed` : ''}`,
      creditsUsed: started,
      creditsRemaining: credits.remaining - started,
      results,
      nextStep: 'GET this endpoint to check progress'
    })
    
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
