import { NextRequest, NextResponse } from 'next/server'
import { heygenClient } from '@/lib/heygen-client'
import { getSql } from '@/lib/db'

interface VideoStatus {
  videoId: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  videoUrl?: string
  thumbnailUrl?: string
  duration?: number
  error?: string
}

/**
 * GET /api/heygen/check-base-videos?ids=id1,id2,id3
 * 
 * Check status of multiple video IDs at once.
 * Returns aggregated status and video URLs for completed videos.
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const idsParam = searchParams.get('ids')
    
    if (!idsParam) {
      return NextResponse.json({
        success: false,
        error: 'Missing required query parameter: ids (comma-separated video IDs)',
      }, { status: 400 })
    }
    
    const videoIds = idsParam.split(',').filter(id => id.trim())
    
    if (videoIds.length === 0) {
      return NextResponse.json({
        success: false,
        error: 'No valid video IDs provided',
      }, { status: 400 })
    }
    
    console.log(`[HeyGen Check] Checking status of ${videoIds.length} videos...`)
    
    // Check all videos in parallel
    const sql = getSql()
    const results = await Promise.allSettled(
      videoIds.map(async (videoId) => {
        const status = await heygenClient.getVideoStatus(videoId)
        
        // Update database with status
        if (status.status === 'completed' && status.video_url) {
          await sql`
            UPDATE heygen_videos SET
              status = 'completed',
              video_url = ${status.video_url},
              thumbnail_url = ${status.thumbnail_url || null},
              duration_seconds = ${status.duration || null},
              updated_at = NOW(),
              completed_at = NOW()
            WHERE heygen_video_id = ${videoId}
          `
        } else if (status.status === 'failed') {
          await sql`
            UPDATE heygen_videos SET
              status = 'failed',
              error_message = ${status.error || 'Unknown error'},
              updated_at = NOW()
            WHERE heygen_video_id = ${videoId}
          `
        }
        
        return {
          videoId,
          status: status.status,
          videoUrl: status.video_url,
          thumbnailUrl: status.thumbnail_url,
          duration: status.duration,
          error: status.error,
        } as VideoStatus
      })
    )
    
    // Process results
    const statuses: VideoStatus[] = results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value
      } else {
        return {
          videoId: videoIds[index],
          status: 'failed' as const,
          error: result.reason?.message || 'Failed to check status',
        }
      }
    })
    
    // Aggregate statistics
    const completed = statuses.filter(s => s.status === 'completed')
    const processing = statuses.filter(s => s.status === 'processing' || s.status === 'pending')
    const failed = statuses.filter(s => s.status === 'failed')
    
    // Generate code snippet for completed videos
    const codeSnippet = completed.length > 0 ? generateCodeSnippet(completed) : null
    
    return NextResponse.json({
      success: true,
      summary: {
        total: statuses.length,
        completed: completed.length,
        processing: processing.length,
        failed: failed.length,
        allDone: processing.length === 0,
      },
      statuses,
      completedVideos: completed.map(s => ({
        videoId: s.videoId,
        videoUrl: s.videoUrl,
        thumbnailUrl: s.thumbnailUrl,
        duration: s.duration,
      })),
      // If still processing, suggest polling again
      ...(processing.length > 0 && {
        suggestion: `${processing.length} videos still processing. Poll again in 10 seconds.`,
        pollAgain: `/api/heygen/check-base-videos?ids=${processing.map(s => s.videoId).join(',')}`,
      }),
      // Code snippet to add to kelly-assets.ts
      ...(codeSnippet && { codeSnippet }),
    })
  } catch (error) {
    console.error('[HeyGen Check] Error:', error)
    
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Status check failed',
    }, { status: 500 })
  }
}

function generateCodeSnippet(completed: VideoStatus[]): string {
  const lines = [
    '// Add to /lib/kelly-assets.ts',
    'export const HEYGEN_BASE_VIDEOS: Record<string, { url: string; thumbnail: string; duration: number }> = {',
  ]
  
  for (const video of completed) {
    if (video.videoUrl) {
      lines.push(`  '${video.videoId}': {`)
      lines.push(`    url: '${video.videoUrl}',`)
      lines.push(`    thumbnail: '${video.thumbnailUrl || ''}',`)
      lines.push(`    duration: ${video.duration || 0},`)
      lines.push(`  },`)
    }
  }
  
  lines.push('};')
  return lines.join('\n')
}
