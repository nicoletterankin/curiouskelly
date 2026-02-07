import { NextRequest, NextResponse } from 'next/server'
import { heygenClient } from '@/lib/heygen-client'

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    
    if (!id) {
      return NextResponse.json(
        { error: 'Video ID is required' },
        { status: 400 }
      )
    }
    
    const status = await heygenClient.getVideoStatus(id)
    
    return NextResponse.json({
      success: true,
      data: status,
    })
  } catch (error) {
    console.error('[HeyGen] Status check error:', error)
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Status check failed',
        success: false,
      },
      { status: 500 }
    )
  }
}
