import { NextRequest, NextResponse } from 'next/server'
import { getAllAvatarConfigs, heygenClient } from '@/lib/heygen-client'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

const BASE_SCRIPT = "Hello! I'm Kelly, your curious learning companion. Every day brings something amazing to discover together!"

export const maxDuration = 300 // 5 minutes max

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'status'
  
  try {
    // Step 1: Always check credits first
    console.log('[HeyGen Run] Checking credits...')
    const credits = await heygenClient.getCredits()
    console.log('[HeyGen Run] Credits available:', credits.remaining)
    
    if (action === 'credits') {
      return NextResponse.json({ 
        credits: credits.remaining,
        message: 'API connection successful'
      })
    }
    
    if (action === 'status') {
      // Get database status
      const sql = getSql()
      const dbStatus = await sql`
        SELECT 
          COUNT(*) FILTER (WHERE status = 'completed') as completed,
          COUNT(*) FILTER (WHERE status = 'processing') as processing,
          COUNT(*) FILTER (WHERE status = 'failed') as failed,
          COUNT(*) as total
        FROM heygen_videos 
        WHERE video_type = 'base'
      `
      
      const stats = dbStatus[0] || { completed: 0, processing: 0, failed: 0, total: 0 }
      
      // Check processing videos
      const processingVideos = await sql`
        SELECT heygen_video_id, age_category, archetype
        FROM heygen_videos 
        WHERE video_type = 'base' AND status = 'processing'
        LIMIT 10
      `
      
      const updates = []
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
                completed_at = NOW()
              WHERE heygen_video_id = ${video.heygen_video_id}
            `
            updates.push({ id: video.heygen_video_id, newStatus: 'completed' })
          } else if (status.status === 'failed') {
            await sql`
              UPDATE heygen_videos SET status = 'failed', error_message = ${status.error || 'Unknown'}
              WHERE heygen_video_id = ${video.heygen_video_id}
            `
            updates.push({ id: video.heygen_video_id, newStatus: 'failed' })
          }
        } catch (e) {
          console.error('[HeyGen Run] Status check error:', e)
        }
      }
      
      return NextResponse.json({
        credits: credits.remaining,
        database: {
          completed: Number(stats.completed),
          processing: Number(stats.processing),
          failed: Number(stats.failed),
          total: Number(stats.total),
          target: 36
        },
        updates,
        nextAction: Number(stats.completed) >= 36 
          ? 'All done! Use action=export to get video URLs'
          : Number(stats.processing) > 0
            ? 'Videos processing. Refresh in 30 seconds.'
            : 'Use action=generate to start video generation'
      })
    }
    
    if (action === 'generate') {
      const limit = parseInt(searchParams.get('limit') || '5', 10)
      const testMode = searchParams.get('test') === 'true'
      
      if (credits.remaining < 1) {
        return NextResponse.json({ error: 'No credits remaining' }, { status: 400 })
      }
      
      // Get existing videos
      const existing = await sql`
        SELECT age_category, archetype FROM heygen_videos 
        WHERE video_type = 'base' AND status IN ('completed', 'processing')
      `
      const existingSet = new Set(existing.map(v => `${v.age_category}-${v.archetype}`))
      
      // Find missing
      const allAvatars = getAllAvatarConfigs()
      const missing = allAvatars.filter(a => !existingSet.has(`${a.ageCategory}-${a.archetype}`))
      
      if (missing.length === 0) {
        return NextResponse.json({ message: 'All 36 avatars already generated or processing' })
      }
      
      const toGenerate = missing.slice(0, Math.min(limit, Math.floor(credits.remaining)))
      const results = []
      
      for (const avatar of toGenerate) {
        try {
          console.log(`[HeyGen Run] Generating ${avatar.ageCategory}_${avatar.archetype}...`)
          console.log(`[HeyGen Run] Avatar config:`, avatar)
          
          const response = await heygenClient.generateVideo({
            avatarGroupId: avatar.avatarGroupId,
            lookId: avatar.lookId,
            script: BASE_SCRIPT,
            title: `Kelly Base - ${avatar.ageCategory} ${avatar.archetype}`,
            test: testMode
          })
          
          await sql`
            INSERT INTO heygen_videos (heygen_video_id, video_type, age_category, archetype, script, status)
            VALUES (${response.video_id}, 'base', ${avatar.ageCategory}, ${avatar.archetype}, ${BASE_SCRIPT}, 'processing')
            ON CONFLICT DO NOTHING
          `
          
          results.push({ 
            avatar: `${avatar.ageCategory}_${avatar.archetype}`, 
            videoId: response.video_id, 
            status: 'started' 
          })
          
          // Delay between requests
          await new Promise(r => setTimeout(r, 1000))
          
        } catch (e) {
          console.error(`[HeyGen Run] Error generating ${avatar.ageCategory}_${avatar.archetype}:`, e)
          results.push({ avatar: `${avatar.ageCategory}_${avatar.archetype}`, error: String(e) })
        }
      }
      
      return NextResponse.json({
        generated: results.filter(r => r.status === 'started').length,
        failed: results.filter(r => r.error).length,
        remaining: missing.length - toGenerate.length,
        results,
        nextAction: 'Use action=status to check progress'
      })
    }
    
    if (action === 'export') {
      const completed = await sql`
        SELECT age_category, archetype, video_url, thumbnail_url, duration_seconds
        FROM heygen_videos 
        WHERE video_type = 'base' AND status = 'completed'
        ORDER BY age_category, archetype
      `
      
      // Generate TypeScript code
      const byAge: Record<string, Record<string, { video_url: string; thumbnail_url: string | null }>> = {}
      for (const v of completed) {
        if (!byAge[v.age_category]) byAge[v.age_category] = {}
        byAge[v.age_category][v.archetype] = {
          video_url: v.video_url,
          thumbnail_url: v.thumbnail_url
        }
      }
      
      return NextResponse.json({
        count: completed.length,
        videos: completed,
        typescript: `// Generated HeyGen Base Videos
export const HEYGEN_BASE_VIDEOS = ${JSON.stringify(byAge, null, 2)} as const;`
      })
    }
    
    return NextResponse.json({ error: 'Unknown action. Use: credits, status, generate, export' }, { status: 400 })
    
  } catch (error) {
    console.error('[HeyGen Run] Error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
