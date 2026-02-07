import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'list'
  
  // Step 1: List avatars to see exact format HeyGen returns
  if (action === 'list') {
    const res = await fetch('https://api.heygen.com/v2/avatars', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    })
    const data = await res.json()
    
    const avatars = data.data?.avatars || []
    
    // Find ANY Kelly-named avatars (check all fields)
    const kellyAvatars = avatars.filter((a: any) => {
      const name = (a.avatar_name || a.name || '').toLowerCase()
      return name.includes('kelly')
    })
    
    // Check for avatars with your known group IDs
    const knownGroupIds = [
      'a762125d3107477aba43d1bd79f90d6e', // adult kelly
      '93bb788b97d847409ad7dcf69702ece5', // kid kelly  
      'd8c4ffac39a546a682b603c56e15906a'  // senior kelly
    ]
    const byGroupId = avatars.filter((a: any) => 
      knownGroupIds.includes(a.avatar_id) || 
      knownGroupIds.includes(a.group_id) ||
      knownGroupIds.includes(a.avatar_group_id)
    )
    
    // Get unique avatar types to understand structure
    const avatarTypes = [...new Set(avatars.map((a: any) => a.type))]
    
    // Get sample of first few avatars to see full structure
    const sampleAvatars = avatars.slice(0, 3)
    
    return NextResponse.json({
      success: res.ok,
      kellyAvatars: kellyAvatars,
      matchedByGroupId: byGroupId,
      avatarTypes: avatarTypes,
      sampleAvatars: sampleAvatars,
      totalCount: avatars.length
    })
  }
  
  // Step 2: Generate video with exact avatar_id from list
  if (action === 'generate') {
    const avatarId = searchParams.get('avatar_id') || '36aadaf237534406b4baae6d850f99c1'
    const script = searchParams.get('script') || 'Hello! I am Kelly, your curious learning companion!'
    
    const payload = {
      video_inputs: [{
        character: {
          type: "avatar",
          avatar_id: avatarId,
          avatar_style: "normal"
        },
        voice: {
          type: "text", 
          input_text: script,
          voice_id: "1bd001e7e50f421d891986aad5158bc8"
        }
      }],
      dimension: { width: 1280, height: 720 },
      test: false
    }
    
    console.log('[HeyGen] Generating with payload:', JSON.stringify(payload, null, 2))
    
    const res = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })
    
    const data = await res.json()
    
    if (res.ok && data.data?.video_id) {
      return NextResponse.json({
        success: true,
        video_id: data.data.video_id,
        checkStatus: `/api/heygen/now?action=status&video_id=${data.data.video_id}`
      })
    }
    
    return NextResponse.json({
      success: false,
      status: res.status,
      error: data,
      triedAvatarId: avatarId
    })
  }
  
  // Step 3: Check video status
  if (action === 'status') {
    const videoId = searchParams.get('video_id')
    if (!videoId) {
      return NextResponse.json({ error: 'Need video_id param' })
    }
    
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    })
    const data = await res.json()
    
    return NextResponse.json({
      success: res.ok,
      status: data.data?.status,
      video_url: data.data?.video_url,
      thumbnail: data.data?.thumbnail_url,
      raw: data
    })
  }
  
  return NextResponse.json({ error: 'Unknown action. Use: list, generate, status' })
}
