import { NextResponse } from 'next/server'
import { heygenClient, getAllAvatarConfigs } from '@/lib/heygen-client'

export async function GET() {
  try {
    console.log('[v0] HeyGen credits - checking API key exists:', !!process.env.HEYGEN_API_KEY)
    console.log('[v0] HeyGen credits - calling getCredits()')
    
    const credits = await heygenClient.getCredits()
    console.log('[v0] HeyGen credits - response:', credits)
    
    const avatarCount = getAllAvatarConfigs().length
    console.log('[v0] HeyGen credits - avatar count:', avatarCount)
    
    return NextResponse.json({
      success: true,
      data: {
        remaining: credits.remaining,
        used: credits.used,
        avatarsConfigured: avatarCount,
        estimatedCostForBaseVideos: avatarCount,
      },
    })
  } catch (error) {
    console.error('[HeyGen] Credits check error:', error)
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Credits check failed',
        success: false,
      },
      { status: 500 }
    )
  }
}
