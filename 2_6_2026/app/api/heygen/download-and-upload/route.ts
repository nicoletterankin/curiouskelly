import { NextRequest, NextResponse } from 'next/server'
import { put } from '@vercel/blob'
import { getSql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

interface VideoMapping {
  video_id: string
  day_number: number
  phase: string
  age_group: string
  archetype: string
}

// GET - List all HeyGen videos with their real URLs
export async function GET() {
  try {
    // Fetch all videos from HeyGen
    const listRes = await fetch('https://api.heygen.com/v1/video.list?limit=100', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY },
    })
    
    if (!listRes.ok) {
      return NextResponse.json({ error: 'Failed to fetch from HeyGen', status: listRes.status }, { status: 500 })
    }
    
    const listData = await listRes.json()
    const videos = listData.data?.videos || []
    
    // Get detailed info for each completed video
    const completedVideos = videos.filter((v: { status: string }) => v.status === 'completed')
    
    const videoDetails = await Promise.all(
      completedVideos.slice(0, 20).map(async (v: { video_id: string }) => {
        try {
          const detailRes = await fetch(
            `https://api.heygen.com/v1/video_status.get?video_id=${v.video_id}`,
            { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
          )
          const detail = await detailRes.json()
          return {
            video_id: v.video_id,
            video_url: detail.data?.video_url || null,
            thumbnail_url: detail.data?.thumbnail_url || null,
            duration: detail.data?.duration || null,
          }
        } catch {
          return { video_id: v.video_id, video_url: null }
        }
      })
    )
    
    return NextResponse.json({
      total: videos.length,
      completed: completedVideos.length,
      videos: videoDetails,
    })
  } catch (error) {
    console.error('Error fetching HeyGen videos:', error)
    return NextResponse.json({ error: 'Failed to fetch videos' }, { status: 500 })
  }
}

// POST - Download from HeyGen, upload to Blob, update DB
export async function POST(request: NextRequest) {
  try {
    const { mappings } = await request.json() as { mappings: VideoMapping[] }
    
    if (!mappings || !Array.isArray(mappings)) {
      return NextResponse.json({ error: 'mappings array required' }, { status: 400 })
    }
    
    const sql = getSql()
    const results = []
    
    for (const mapping of mappings) {
      try {
        // 1. Get video URL from HeyGen
        const statusRes = await fetch(
          `https://api.heygen.com/v1/video_status.get?video_id=${mapping.video_id}`,
          { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
        )
        const statusData = await statusRes.json()
        const heygenUrl = statusData.data?.video_url
        
        if (!heygenUrl) {
          results.push({ video_id: mapping.video_id, success: false, error: 'No video URL from HeyGen' })
          continue
        }
        
        // 2. Download video from HeyGen
        const videoRes = await fetch(heygenUrl)
        if (!videoRes.ok) {
          results.push({ video_id: mapping.video_id, success: false, error: 'Failed to download video' })
          continue
        }
        const videoBlob = await videoRes.blob()
        
        // 3. Upload to Vercel Blob
        const blobResult = await put(
          `kelly-videos/${mapping.video_id}.mp4`,
          videoBlob,
          { access: 'public', contentType: 'video/mp4' }
        )
        
        // 4. Update database
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${blobResult.url}, status = 'completed', updated_at = NOW()
          WHERE day_number = ${mapping.day_number}
            AND phase = ${mapping.phase}
            AND age_group = ${mapping.age_group}
            AND archetype = ${mapping.archetype}
        `
        
        // Also update heygen_videos table
        await sql`
          INSERT INTO heygen_videos (heygen_video_id, video_url, status, day_of_year, phase, age_category, archetype, created_at)
          VALUES (${mapping.video_id}, ${blobResult.url}, 'completed', ${mapping.day_number}, ${mapping.phase}, ${mapping.age_group}, ${mapping.archetype}, NOW())
          ON CONFLICT (day_of_year, phase, age_category, archetype) 
          DO UPDATE SET video_url = ${blobResult.url}, heygen_video_id = ${mapping.video_id}, status = 'completed'
        `
        
        results.push({ 
          video_id: mapping.video_id, 
          success: true, 
          blob_url: blobResult.url,
          mapping 
        })
      } catch (err) {
        results.push({ video_id: mapping.video_id, success: false, error: String(err) })
      }
    }
    
    return NextResponse.json({ 
      processed: results.length,
      successful: results.filter(r => r.success).length,
      results 
    })
  } catch (error) {
    console.error('Error processing videos:', error)
    return NextResponse.json({ error: 'Failed to process videos' }, { status: 500 })
  }
}
