import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Your custom Kelly Photo Avatar look_ids (the look_id IS the talking_photo_id)
const KELLY_LOOKS = {
  kid: {
    storyteller: '1024bc304a1146998bc4c360173b2c48',
    scientist: '82813816115c4fbe93b3f3f211bd9931',
    explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
  },
  adult: {
    storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
    scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
    explorer: '62516885ca4b4eae8f63b87b8c060e25',
  },
  senior: {
    storyteller: '98178c87897e4421884b535b7864ba86',
    scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
    explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const videoId = searchParams.get('video_id')
  
  // If video_id provided, check status
  if (videoId) {
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    })
    const data = await res.json()
    return NextResponse.json({
      videoId,
      status: data.data?.status,
      videoUrl: data.data?.video_url,
      thumbnailUrl: data.data?.thumbnail_url,
      duration: data.data?.duration,
      error: data.data?.error
    })
  }
  
  // Generate new video using YOUR custom Kelly Photo Avatar
  const age = (searchParams.get('age') || 'adult') as 'kid' | 'adult' | 'senior'
  const archetype = searchParams.get('archetype') || 'storyteller'
  const talkingPhotoId = KELLY_LOOKS[age]?.[archetype as keyof typeof KELLY_LOOKS['adult']] || KELLY_LOOKS.adult.storyteller
  
  const script = "Hello! I am Kelly, your curious learning companion. Today we are going to explore why the sky is blue. It is because of something called light scattering!"

  console.log('[HeyGen] Using talking_photo type for custom Kelly avatar')
  console.log('[HeyGen] talking_photo_id:', talkingPhotoId)

  // CORRECT PAYLOAD for Photo Avatars (your custom Kelly)
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',           // NOT "avatar"!
        talking_photo_id: talkingPhotoId // The look_id IS the talking_photo_id
      },
      voice: {
        type: 'text',
        input_text: script,
        voice_id: '1bd001e7e50f421d891986aad5158bc8'
      },
      background: {
        type: 'color',
        value: '#FFFFFF'
      }
    }],
    dimension: {
      width: 1920,
      height: 1080
    },
    test: false  // Use real credits
  }

  const res = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })

  const data = await res.json()

  if (!res.ok) {
    return NextResponse.json({
      success: false,
      error: data,
      payload,
      hint: 'Using talking_photo type with your custom Kelly look_id'
    }, { status: res.status })
  }

  const newVideoId = data.data?.video_id

  return NextResponse.json({
    success: true,
    videoId: newVideoId,
    message: 'Video generation started with YOUR custom Kelly!',
    checkStatus: `/api/heygen/trigger?video_id=${newVideoId}`,
    talkingPhotoId,
    age,
    archetype
  })
}
