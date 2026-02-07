import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_BASE = 'https://api.heygen.com'

// Kelly's avatar IDs from kelly-assets.ts - using adult storyteller as default test
const KELLY_AVATAR_GROUP_ID = 'a762125d3107477aba43d1bd79f90d6e' // adult Kelly group
const KELLY_LOOK_ID = '3d6a9d6f91b444469dae87ebb3d9eba6' // adult storyteller look

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'credits'
  
  const apiKey = process.env.HEYGEN_API_KEY
  
  // Debug info
  const debug = {
    apiKeySet: !!apiKey,
    apiKeyPrefix: apiKey ? apiKey.substring(0, 15) + '...' : 'NOT SET',
    avatarGroupId: KELLY_AVATAR_GROUP_ID,
    lookId: KELLY_LOOK_ID,
    action
  }
  
  if (!apiKey) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not set', debug })
  }

  try {
    if (action === 'credits') {
      const res = await fetch(`${HEYGEN_API_BASE}/v2/user/remaining_quota`, {
        headers: { 'X-Api-Key': apiKey }
      })
      const data = await res.json()
      return NextResponse.json({ 
        success: res.ok,
        status: res.status,
        credits: data,
        debug
      })
    }
    
    if (action === 'avatars') {
      const res = await fetch(`${HEYGEN_API_BASE}/v2/avatars`, {
        headers: { 'X-Api-Key': apiKey }
      })
      const data = await res.json()
      const avatars = data.data?.avatars || []
      
      // Find Kelly avatars specifically
      const kellyAvatars = avatars.filter((a: { avatar_name?: string }) => 
        a.avatar_name?.toLowerCase().includes('kelly')
      )
      
      return NextResponse.json({ 
        success: res.ok,
        count: avatars.length,
        kellyAvatars: kellyAvatars,
        firstFewAvatars: avatars.slice(0, 5),
        debug
      })
    }
    
    if (action === 'generate') {
      const script = searchParams.get('script') || 
        "Hello! I am Kelly, your curious learning companion. Today we are going to explore something amazing together!"
      
      // For custom instant avatars with looks, use avatar_id as group#look format
      // or try different formats based on HeyGen API requirements
      const avatarIdFormat = searchParams.get('format') || 'combined'
      
      let avatarId: string
      if (avatarIdFormat === 'group') {
        avatarId = KELLY_AVATAR_GROUP_ID
      } else if (avatarIdFormat === 'look') {
        avatarId = KELLY_LOOK_ID
      } else {
        // Combined format: group_id#look_id
        avatarId = `${KELLY_AVATAR_GROUP_ID}#${KELLY_LOOK_ID}`
      }
      
      // For instant avatars with looks, use avatar_id (group) + look_id
      const payload = {
        video_inputs: [{
          character: {
            type: "avatar",
            avatar_id: KELLY_AVATAR_GROUP_ID,
            avatar_style: "normal",
            look_id: KELLY_LOOK_ID
          },
          voice: {
            type: "text",
            input_text: script,
            voice_id: "1bd001e7e50f421d891986aad5158bc8"
          }
        }],
        dimension: {
          width: 1280,
          height: 720
        },
        test: false  // Use real credits, not test mode
      }
      
      const res = await fetch(`${HEYGEN_API_BASE}/v2/video/generate`, {
        method: 'POST',
        headers: {
          'X-Api-Key': apiKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })
      
      const data = await res.json()
      return NextResponse.json({
        success: res.ok,
        status: res.status,
        response: data,
        debug,
        nextStep: data.data?.video_id 
          ? `/api/heygen/simple?action=status&video_id=${data.data.video_id}`
          : null
      })
    }
    
    if (action === 'status') {
      const videoId = searchParams.get('video_id') || searchParams.get('id')
      if (!videoId) {
        return NextResponse.json({ error: 'Missing video_id parameter' })
      }
      
      const res = await fetch(`${HEYGEN_API_BASE}/v1/video_status.get?video_id=${videoId}`, {
        headers: { 'X-Api-Key': apiKey }
      })
      const data = await res.json()
      return NextResponse.json({
        success: res.ok,
        videoStatus: data.data?.status,
        videoUrl: data.data?.video_url,
        full: data,
        debug
      })
    }
    
    return NextResponse.json({ 
      error: 'Unknown action',
      availableActions: ['credits', 'avatars', 'generate', 'status'],
      debug
    })
    
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error',
      debug
    })
  }
}
