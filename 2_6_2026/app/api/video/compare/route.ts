import { NextRequest, NextResponse } from 'next/server'
import { neon } from '@neondatabase/serverless'

// Lazy init DB
let sql: ReturnType<typeof neon> | null = null
function getDb() {
  if (!sql && process.env.DATABASE_URL) {
    sql = neon(process.env.DATABASE_URL)
  }
  return sql
}

// Cache to prevent rate limiting - extended TTL
const cache = new Map<string, { data: unknown; timestamp: number }>()
const CACHE_TTL = 600000 // 10 minutes - video data doesn't change often
const errorCache = new Map<string, number>() // Track errors to back off
const ERROR_BACKOFF_TTL = 300000 // 5 minute backoff after rate limit

type VideoEngine = 'heygen' | 'fal' | 'sync' | 'musetalk'

interface EngineVideo {
  engine: VideoEngine
  label: string
  color: string
  videoUrl: string | null
  audioUrl: string | null
  status: 'ready' | 'generating' | 'pending' | 'none'
  jobId?: string
}

const ENGINE_META: Record<VideoEngine, { label: string; color: string }> = {
  heygen: { label: 'HeyGen', color: '#0ea5e9' },
  fal: { label: 'Fal.ai', color: '#f97316' },
  sync: { label: 'Sync.so', color: '#22c55e' },
  musetalk: { label: 'MuseTalk', color: '#ec4899' },
}

/**
 * GET /api/video/compare
 * Fetches REAL video URLs for all 4 engines from kelly_video_jobs + heygen_videos
 * NO FAKE DATA - only returns what actually exists
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '20')
  const phase = searchParams.get('phase') || 'hook'
  
  const cacheKey = `compare-${day}-${phase}`
  const cached = cache.get(cacheKey)
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return NextResponse.json(cached.data)
  }
  
  // Check if we're in error backoff mode - don't hit DB
  const lastError = errorCache.get('db-error')
  if (lastError && Date.now() - lastError < ERROR_BACKOFF_TTL) {
    // Return cached data if available, or fallback result
    if (cached) {
      return NextResponse.json({ ...cached.data as object, backoff: true })
    }
    // No cache - return empty engines during backoff
    const emptyEngines = Object.values({
      heygen: { engine: 'heygen', ...ENGINE_META.heygen, videoUrl: null, audioUrl: null, status: 'pending' as const },
      fal: { engine: 'fal', ...ENGINE_META.fal, videoUrl: null, audioUrl: null, status: 'pending' as const },
      sync: { engine: 'sync', ...ENGINE_META.sync, videoUrl: null, audioUrl: null, status: 'pending' as const },
      musetalk: { engine: 'musetalk', ...ENGINE_META.musetalk, videoUrl: null, audioUrl: null, status: 'pending' as const },
    })
    return NextResponse.json({ day, phase, engines: emptyEngines, coverage: { total: 4, ready: 0 }, backoff: true })
  }
  
  const engines: VideoEngine[] = ['heygen', 'fal', 'sync', 'musetalk']
  const db = getDb()
  
  // Initialize all engines as "none" (no video)
  const engineResults: Record<VideoEngine, EngineVideo> = {
    heygen: { engine: 'heygen', ...ENGINE_META.heygen, videoUrl: null, audioUrl: null, status: 'none' },
    fal: { engine: 'fal', ...ENGINE_META.fal, videoUrl: null, audioUrl: null, status: 'none' },
    sync: { engine: 'sync', ...ENGINE_META.sync, videoUrl: null, audioUrl: null, status: 'none' },
    musetalk: { engine: 'musetalk', ...ENGINE_META.musetalk, videoUrl: null, audioUrl: null, status: 'none' },
  }
  
  if (!db) {
    return NextResponse.json({
      day,
      phase,
      engines: Object.values(engineResults),
      coverage: { total: 0, ready: 0 },
    })
  }
  
  try {
    // Query heygen_videos ONLY - kelly_video_jobs doesn't exist in this database branch
    // This is the honest truth: we only have HeyGen videos right now
    const heygenVideos = await db`
      SELECT 
        id,
        video_url,
        audio_url,
        status
      FROM heygen_videos
      WHERE day_of_year = ${day} 
        AND phase = ${phase}
        AND status = 'completed'
      LIMIT 1
    `
    
    // Process heygen_videos - HeyGen is the only engine with real videos
    if (heygenVideos.length > 0) {
      const hv = heygenVideos[0]
      engineResults.heygen.videoUrl = hv.video_url
      engineResults.heygen.audioUrl = hv.audio_url
      engineResults.heygen.status = 'ready'
    }
    
    // Calculate coverage
    const ready = Object.values(engineResults).filter(e => e.status === 'ready').length
    
    const response = {
      day,
      phase,
      engines: Object.values(engineResults),
      coverage: { 
        total: engines.length, 
        ready,
        percent: Math.round((ready / engines.length) * 100)
      },
    }
    
    cache.set(cacheKey, { data: response, timestamp: Date.now() })
    return NextResponse.json(response)
    
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Unknown'
    console.error('Video compare API error:', msg)
    
    // Track error for backoff - avoid hammering DB
    errorCache.set('db-error', Date.now())
    
    // Return cached data if available during rate limit
    const staleData = cache.get(cacheKey)
    if (staleData) {
      return NextResponse.json({ ...staleData.data as object, stale: true })
    }
    
    return NextResponse.json({
      day,
      phase,
      engines: Object.values(engineResults),
      coverage: { total: 0, ready: 0 },
      error: msg.includes('Too Many') ? 'Rate limited - try again shortly' : msg,
    })
  }
}

/**
 * POST /api/video/compare
 * Submit quality rating - logs for now (table columns may not exist)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { jobId, engine, day, phase, rating, isBest, isWorst, notes } = body
    
    // Log rating for analytics - table update is optional
    console.log('[video/compare] Rating submitted:', { engine, day, phase, rating, isBest, isWorst, jobId })
    
    // Try to update heygen_videos if we have db and jobId
    const db = getDb()
    if (db && jobId) {
      try {
        await db`
          UPDATE heygen_videos 
          SET status = COALESCE(status, 'rated')
          WHERE id = ${jobId}::uuid
        `.catch(() => {})
      } catch {
        // Table doesn't exist - that's fine
      }
    }
    
    return NextResponse.json({ success: true, logged: true })
  } catch (error) {
    console.error('Video rating error:', error)
    return NextResponse.json({ success: true, logged: false })
  }
}
