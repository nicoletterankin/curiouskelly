/**
 * HEYGEN VIDEO STATUS CHECK
 * 
 * GET /api/heygen/status?videoId=xxx
 * GET /api/heygen/status?video_id=xxx (legacy)
 * 
 * Returns the current status of a HeyGen video generation job.
 * Status values: pending, processing, completed, failed
 */

import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_BASE = 'https://api.heygen.com'

export async function GET(request: NextRequest) {
  const apiKey = process.env.HEYGEN_API_KEY
  if (!apiKey) {
    return NextResponse.json({ 
      error: 'HEYGEN_API_KEY not set',
      fix: 'Add HEYGEN_API_KEY to environment variables'
    }, { status: 500 })
  }

  const { searchParams } = new URL(request.url)
  const videoId = searchParams.get('videoId') || searchParams.get('video_id')
  
  if (!videoId) {
    return NextResponse.json({ 
      error: 'videoId parameter required',
      usage: 'GET /api/heygen/status?videoId=YOUR_VIDEO_ID'
    }, { status: 400 })
  }

  try {
    console.log(`[heygen/status] Checking video: ${videoId}`)
    
    const res = await fetch(`${HEYGEN_API_BASE}/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': apiKey }
    })
    
    const responseText = await res.text()
    console.log(`[heygen/status] Response ${res.status}:`, responseText.substring(0, 500))
    
    let data: Record<string, unknown>
    try {
      data = JSON.parse(responseText)
    } catch {
      data = { rawText: responseText }
    }

    if (!res.ok) {
      return NextResponse.json({
        error: 'HeyGen API Error',
        httpStatus: res.status,
        videoId,
        response: data
      }, { status: res.status })
    }

    const statusData = data.data as Record<string, unknown> | undefined
    const status = statusData?.status as string || 'unknown'

    return NextResponse.json({
      videoId,
      status,
      isComplete: status === 'completed',
      isFailed: status === 'failed',
      isProcessing: status === 'processing' || status === 'pending',
      videoUrl: statusData?.video_url || null,
      thumbnailUrl: statusData?.thumbnail_url || null,
      duration: statusData?.duration || null,
      error: statusData?.error || null,
      raw: statusData
    })
    
  } catch (error) {
    console.error('[heygen/status] Exception:', error)
    return NextResponse.json({
      error: 'Request failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      videoId
    }, { status: 500 })
  }
}
