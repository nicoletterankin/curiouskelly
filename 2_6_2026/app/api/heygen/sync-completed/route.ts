import { NextRequest, NextResponse } from 'next/server'
import { put } from '@vercel/blob'
import { sql } from '@/lib/db'
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

export const maxDuration = 300

/**
 * GET - List all completed HeyGen videos and their sync status
 * POST - Download completed videos from HeyGen, upload to Vercel Blob, update DB
 */

interface HeyGenVideo {
  video_id: string
  status: string
  video_title?: string
  created_at?: number
}

interface HeyGenVideoDetail {
  video_id: string
  status: string
  video_url?: string
  thumbnail_url?: string
  duration?: number
}

// Parse video title to extract metadata: day21_de_senior_macgyver_action
function parseVideoTitle(title: string): {
  day: number
  language: string
  age: string
  archetype: string
  phase: string
} | null {
  // Pattern: day{N}_{lang}_{age}_{archetype}_{phase}
  const match = title.match(/day(\d+)_([a-z]{2})_(\w+)_(\w+)_(\w+)/i)
  if (!match) return null
  
  return {
    day: parseInt(match[1]),
    language: match[2].toLowerCase(),
    age: match[3].toLowerCase(),
    archetype: match[4].toLowerCase(),
    phase: match[5].toLowerCase()
  }
}

export async function GET(request: NextRequest) {
  // HeyGen API max limit is 100
  const requestedLimit = parseInt(request.nextUrl.searchParams.get('limit') || '100')
  const limit = Math.min(requestedLimit, 100)
  
  try {
    // Fetch all videos from HeyGen
    const listRes = await fetch(`https://api.heygen.com/v1/video.list?limit=${limit}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY },
    })
    
    if (!listRes.ok) {
      return NextResponse.json({ error: 'Failed to fetch from HeyGen', status: listRes.status }, { status: 500 })
    }
    
    const listData = await listRes.json()
    const videos: HeyGenVideo[] = listData.data?.videos || []
    
    const completedVideos = videos.filter(v => v.status === 'completed')
    
    // Check which are already synced
    const syncedIds = await sql`
      SELECT heygen_video_id FROM heygen_videos 
      WHERE heygen_video_id IS NOT NULL 
        AND video_url LIKE 'https://%.blob.vercel-storage.com/%'
        AND status = 'completed'
    `
    const syncedSet = new Set(syncedIds.map(r => r.heygen_video_id))
    
    const needsSync = completedVideos.filter(v => !syncedSet.has(v.video_id))
    
    return NextResponse.json({
      total: videos.length,
      completed: completedVideos.length,
      alreadySynced: syncedSet.size,
      needsSync: needsSync.length,
      videos: needsSync.slice(0, 20).map(v => ({
        video_id: v.video_id,
        title: v.video_title,
        parsed: parseVideoTitle(v.video_title || '')
      }))
    })
  } catch (error) {
    console.error('Error fetching HeyGen videos:', error)
    return NextResponse.json({ error: 'Failed to fetch videos' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}))
  const batchSize = Math.min(body.batchSize || 10, 50)
  
  try {
    // 1. Get list of completed videos from HeyGen (max 100 per HeyGen API)
    const listRes = await fetch('https://api.heygen.com/v1/video.list?limit=100', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY },
    })
    
    if (!listRes.ok) {
      return NextResponse.json({ error: 'Failed to fetch from HeyGen' }, { status: 500 })
    }
    
    const listData = await listRes.json()
    const videos: HeyGenVideo[] = listData.data?.videos || []
    const completedVideos = videos.filter(v => v.status === 'completed')
    
    // 2. Check which are already synced to Blob
    const syncedIds = await sql`
      SELECT heygen_video_id FROM heygen_videos 
      WHERE heygen_video_id IS NOT NULL 
        AND video_url LIKE 'https://%.blob.vercel-storage.com/%'
        AND status = 'completed'
    `
    const syncedSet = new Set(syncedIds.map(r => r.heygen_video_id))
    
    // 3. Process videos that need syncing
    const needsSync = completedVideos.filter(v => !syncedSet.has(v.video_id))
    const toProcess = needsSync.slice(0, batchSize)
    
    const results: Array<{
      video_id: string
      success: boolean
      blob_url?: string
      parsed?: ReturnType<typeof parseVideoTitle>
      error?: string
    }> = []
    
    for (const video of toProcess) {
      try {
        // Get video details including URL
        const detailRes = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${video.video_id}`,
          { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
        )
        const detail: { data?: HeyGenVideoDetail } = await detailRes.json()
        
        if (!detail.data?.video_url) {
          results.push({ video_id: video.video_id, success: false, error: 'No video URL' })
          continue
        }
        
        // Download video
        const videoRes = await fetch(detail.data.video_url)
        if (!videoRes.ok) {
          results.push({ video_id: video.video_id, success: false, error: 'Download failed' })
          continue
        }
        const videoBlob = await videoRes.blob()
        
        // Upload to Vercel Blob
        const blobResult = await put(
          `kelly-videos/${video.video_id}.mp4`,
          videoBlob,
          { access: 'public', contentType: 'video/mp4' }
        )
        
        // Parse title for metadata
        const parsed = parseVideoTitle(video.video_title || '')
        
        if (parsed) {
          // Update heygen_videos table
          await sql`
            INSERT INTO heygen_videos (
              id, day_of_year, phase, age_category, archetype,
              heygen_video_id, video_url, status, duration_seconds, created_at, completed_at
            ) VALUES (
              gen_random_uuid(),
              ${parsed.day}, ${parsed.phase}, ${parsed.age}, ${parsed.archetype},
              ${video.video_id}, ${blobResult.url}, 'completed', ${detail.data.duration || null}, NOW(), NOW()
            )
            ON CONFLICT (day_of_year, phase, age_category, archetype) 
            DO UPDATE SET 
              video_url = ${blobResult.url},
              heygen_video_id = ${video.video_id},
              status = 'completed',
              duration_seconds = ${detail.data.duration || null},
              completed_at = NOW()
          `
          
          // Also update kelly_lesson_assets
          await sql`
            INSERT INTO kelly_lesson_assets (
              id, day_number, phase, age_group, archetype, language,
              video_url, status, created_at
            ) VALUES (
              gen_random_uuid(),
              ${parsed.day}, ${parsed.phase}, ${parsed.age}, ${parsed.archetype}, ${parsed.language},
              ${blobResult.url}, 'completed', NOW()
            )
            ON CONFLICT (day_number, phase, age_group, archetype, language)
            DO UPDATE SET video_url = ${blobResult.url}, status = 'completed'
          `.catch(e => console.warn('kelly_lesson_assets upsert:', e.message))
        }
        
        results.push({
          video_id: video.video_id,
          success: true,
          blob_url: blobResult.url,
          parsed
        })
        
        // Small delay to avoid rate limits
        await new Promise(r => setTimeout(r, 500))
        
      } catch (err) {
        results.push({
          video_id: video.video_id,
          success: false,
          error: err instanceof Error ? err.message : String(err)
        })
      }
    }
    
    return NextResponse.json({
      processed: results.length,
      successful: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length,
      remaining: needsSync.length - toProcess.length,
      results
    })
    
  } catch (error) {
    console.error('Error syncing videos:', error)
    return NextResponse.json({ error: 'Failed to sync videos' }, { status: 500 })
  }
}
