import { NextRequest, NextResponse } from 'next/server'

// Simple HeyGen test endpoint - no database dependencies
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'credits'
  
  const apiKey = process.env.HEYGEN_API_KEY
  
  if (!apiKey) {
    return NextResponse.json({ 
      error: 'HEYGEN_API_KEY not configured',
      keyExists: false 
    }, { status: 500 })
  }
  
  console.log('[v0] HeyGen test - API key exists:', !!apiKey)
  console.log('[v0] HeyGen test - API key length:', apiKey.length)
  console.log('[v0] HeyGen test - action:', action)
  
  try {
    if (action === 'credits') {
      // Check remaining credits
      const response = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
        method: 'GET',
        headers: {
          'X-Api-Key': apiKey,
        },
      })
      
      console.log('[v0] HeyGen credits response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.log('[v0] HeyGen credits error:', errorText)
        return NextResponse.json({ 
          error: `HeyGen API error: ${response.status}`,
          details: errorText 
        }, { status: response.status })
      }
      
      const data = await response.json()
      console.log('[v0] HeyGen credits data:', data)
      
      return NextResponse.json({
        success: true,
        credits: data.data?.remaining_quota ?? data.remaining ?? 'unknown',
        raw: data
      })
    }
    
    if (action === 'generate') {
      // Generate a test video with kid scientist
      const avatarGroupId = '93bb788b97d847409ad7dcf69702ece5' // kid
      const lookId = '82813816115c4fbe93b3f3f211bd9931' // scientist
      
      const payload = {
        video_inputs: [
          {
            character: {
              type: 'avatar',
              avatar_id: avatarGroupId,
              avatar_style: 'normal',
              look_id: lookId,
            },
            voice: {
              type: 'text',
              input_text: 'Hello! I am Kelly, your curious learning companion. Today we are going to explore something amazing together!',
              voice_id: 'en-US-JennyNeural',
            },
          },
        ],
        dimension: {
          width: 1920,
          height: 1080,
        },
        test: true, // Use test mode for faster generation
      }
      
      console.log('[v0] HeyGen generate payload:', JSON.stringify(payload, null, 2))
      
      const response = await fetch('https://api.heygen.com/v2/video/generate', {
        method: 'POST',
        headers: {
          'X-Api-Key': apiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })
      
      console.log('[v0] HeyGen generate response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.log('[v0] HeyGen generate error:', errorText)
        return NextResponse.json({ 
          error: `HeyGen API error: ${response.status}`,
          details: errorText 
        }, { status: response.status })
      }
      
      const data = await response.json()
      console.log('[v0] HeyGen generate data:', data)
      
      return NextResponse.json({
        success: true,
        videoId: data.data?.video_id,
        status: data.data?.status,
        raw: data,
        nextStep: `GET /api/heygen/test?action=status&id=${data.data?.video_id}`
      })
    }
    
    if (action === 'status') {
      const videoId = searchParams.get('id')
      if (!videoId) {
        return NextResponse.json({ error: 'Missing video id parameter' }, { status: 400 })
      }
      
      const response = await fetch(`https://api.heygen.com/v2/video_status.get?video_id=${videoId}`, {
        method: 'GET',
        headers: {
          'X-Api-Key': apiKey,
        },
      })
      
      console.log('[v0] HeyGen status response:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        return NextResponse.json({ 
          error: `HeyGen API error: ${response.status}`,
          details: errorText 
        }, { status: response.status })
      }
      
      const data = await response.json()
      console.log('[v0] HeyGen status data:', data)
      
      return NextResponse.json({
        success: true,
        videoId,
        status: data.data?.status,
        videoUrl: data.data?.video_url,
        thumbnailUrl: data.data?.thumbnail_url,
        duration: data.data?.duration,
        raw: data
      })
    }
    
    return NextResponse.json({
      error: 'Unknown action',
      validActions: ['credits', 'generate', 'status']
    }, { status: 400 })
    
  } catch (error) {
    console.error('[v0] HeyGen test error:', error)
    return NextResponse.json({ 
      error: 'Request failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}
