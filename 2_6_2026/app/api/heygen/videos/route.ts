import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

/**
 * GET /api/heygen/videos - List all generated videos with status
 * GET /api/heygen/videos?filter=day18 - Filter for Day 18 videos only
 * GET /api/heygen/videos?video_id=xxx - Get single video details
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const videoId = searchParams.get('video_id')
  const filter = searchParams.get('filter')
  
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  // Single video status
  if (videoId) {
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    })
    const data = await res.json()
    return NextResponse.json(data)
  }

  // List all videos
  const res = await fetch('https://api.heygen.com/v1/video.list?limit=200', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  })
  const data = await res.json()
  
  if (!res.ok) {
    return NextResponse.json({ error: 'Failed to fetch videos', details: data }, { status: res.status })
  }

  const videos = data.data?.videos || []
  
  // Extract video info
  const processedVideos = videos.map((v: any) => ({
    id: v.video_id,
    status: v.status,
    duration: v.duration,
    thumbnail: v.thumbnail_url,
    videoUrl: v.video_url,
    createdAt: v.created_at,
  }))

  // Summary by status
  const summary = {
    total: processedVideos.length,
    completed: processedVideos.filter((v: any) => v.status === 'completed').length,
    processing: processedVideos.filter((v: any) => v.status === 'processing').length,
    failed: processedVideos.filter((v: any) => v.status === 'failed').length,
  }

  return NextResponse.json({
    success: true,
    summary,
    videos: processedVideos.slice(0, 50), // First 50
    recentCompleted: processedVideos
      .filter((v: any) => v.status === 'completed' && v.videoUrl)
      .slice(0, 10)
  })
}
