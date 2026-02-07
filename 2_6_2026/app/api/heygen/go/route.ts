import { NextRequest, NextResponse } from 'next/server'
import { getKellyAvatar } from '@/lib/kelly-avatars'

const HEYGEN_API = 'https://api.heygen.com/v2'

export async function GET(request: NextRequest) {
  const apiKey = process.env.HEYGEN_API_KEY
  if (!apiKey) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not set' })
  }
  
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action') || 'generate'
  
  // Check credits
  if (action === 'credits') {
    const res = await fetch(`${HEYGEN_API}/user/remaining_quota`, {
      headers: { 'X-Api-Key': apiKey }
    })
    return NextResponse.json(await res.json())
  }
  
  // Check video status
  if (action === 'status') {
    const videoId = searchParams.get('video_id')
    if (!videoId) return NextResponse.json({ error: 'Need video_id param' })
    
    const res = await fetch(`${HEYGEN_API}/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': apiKey }
    })
    return NextResponse.json(await res.json())
  }
  
  // GENERATE VIDEO
  const age = parseInt(searchParams.get('age') || '25')
  const archetype = (searchParams.get('archetype') || 'storyteller') as 'storyteller'
  const script = searchParams.get('script') || 
    "Hello! I am Kelly, your curious learning companion. Today we are going to explore something amazing together! Are you ready?"
  
  const { avatar_id, look_id, age_group } = getKellyAvatar(age, archetype)
  
  const payload = {
    video_inputs: [{
      character: {
        type: 'avatar',
        avatar_id: avatar_id,
        avatar_style: 'normal'
      },
      voice: {
        type: 'text',
        input_text: script,
        voice_id: '1bd001e7e50f421d891986aad5158bc8'
      }
    }],
    dimension: { width: 1280, height: 720 },
    test: false
  }
  
  console.log('[HeyGen GO] Generating video with:', { avatar_id, look_id, age_group, archetype })
  console.log('[HeyGen GO] Payload:', JSON.stringify(payload, null, 2))
  
  const res = await fetch(`${HEYGEN_API}/video/generate`, {
    method: 'POST',
    headers: {
      'X-Api-Key': apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })
  
  const data = await res.json()
  
  if (!res.ok) {
    return NextResponse.json({
      error: 'HeyGen API error',
      status: res.status,
      response: data,
      tried: { avatar_id, look_id, age_group, archetype }
    })
  }
  
  const videoId = data.data?.video_id
  
  return NextResponse.json({
    success: true,
    video_id: videoId,
    message: 'Video generation started! Check status in 30-60 seconds.',
    check_status: `/api/heygen/go?action=status&video_id=${videoId}`,
    avatar_used: { avatar_id, look_id, age_group, archetype }
  })
}
