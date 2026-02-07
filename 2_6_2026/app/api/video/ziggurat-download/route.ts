import { list } from '@vercel/blob'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    // List all blobs with the ziggurat prefix
    const { blobs } = await list({ prefix: 'video/ziggurat/' })
    
    // Filter and format the segments
    const segments = blobs
      .filter(blob => blob.pathname.includes('segment_') && blob.pathname.endsWith('.mp3'))
      .map(blob => {
        const match = blob.pathname.match(/segment_(\d+)\.mp3/)
        return {
          id: match ? match[1] : 'unknown',
          url: blob.url,
          size: blob.size,
          uploaded: blob.uploadedAt
        }
      })
      .sort((a, b) => a.id.localeCompare(b.id))
    
    return NextResponse.json({
      count: segments.length,
      segments,
      // Direct download links for convenience
      allUrls: segments.map(s => s.url)
    })
  } catch (error) {
    return NextResponse.json({ 
      error: 'Failed to list blobs',
      message: (error as Error).message 
    }, { status: 500 })
  }
}
