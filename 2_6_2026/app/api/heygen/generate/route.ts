import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'EXAVITQu4vr4xnSDxMaL' // Default female voice

// Kelly avatar IDs from HeyGen (string names, not UUIDs)
const KELLY_AVATARS = {
  default: 'Kelly_Blue_Shirt_Front',
  blueShirt: 'Kelly_Blue_Shirt_Front',
  doctor: 'Kelly_Doctor_Front',
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { 
      script, 
      avatarId = KELLY_AVATARS.default,
      useElevenLabs = true 
    } = body

    if (!script) {
      return NextResponse.json({ error: 'Script is required' }, { status: 400 })
    }

    let audioUrl: string | null = null

    // Step 1: Generate audio with ElevenLabs (optional but recommended)
    if (useElevenLabs && ELEVENLABS_API_KEY) {
      console.log('[HeyGen] Generating ElevenLabs audio...')
      
      const elevenLabsRes = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
        {
          method: 'POST',
          headers: {
            'xi-api-key': ELEVENLABS_API_KEY,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: script,
            model_id: 'eleven_monolingual_v1',
            voice_settings: {
              stability: 0.5,
              similarity_boost: 0.75,
            },
          }),
        }
      )

      if (elevenLabsRes.ok) {
        // Upload audio to a temporary URL or use blob
        const audioBlob = await elevenLabsRes.blob()
        // For now, we'll use HeyGen's built-in TTS since we need a public URL for audio
        console.log('[HeyGen] ElevenLabs audio generated, but using HeyGen TTS for simplicity')
      }
    }

    // Step 2: Generate video with HeyGen
    console.log('[HeyGen] Generating video with avatar:', avatarId)

    const heygenPayload = {
      video_inputs: [{
        character: {
          type: 'avatar',
          avatar_id: avatarId,
          avatar_style: 'normal'
        },
        voice: {
          type: 'text',
          input_text: script,
          voice_id: '1bd001e7e50f421d891986aad5158bc8' // HeyGen female voice
        }
      }],
      dimension: {
        width: 1280,
        height: 720
      },
      test: false // Use real credits
    }

    const heygenRes = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(heygenPayload)
    })

    const heygenData = await heygenRes.json()

    if (!heygenRes.ok) {
      console.error('[HeyGen] Error:', heygenData)
      return NextResponse.json({ 
        error: 'HeyGen API error', 
        details: heygenData,
        payload: heygenPayload
      }, { status: heygenRes.status })
    }

    const videoId = heygenData.data?.video_id

    return NextResponse.json({
      success: true,
      videoId,
      status: 'processing',
      checkStatusUrl: `/api/heygen/generate?video_id=${videoId}`,
      message: 'Video generation started! Check status in 1-2 minutes.'
    })

  } catch (error) {
    console.error('[HeyGen] Error:', error)
    return NextResponse.json({ 
      error: 'Failed to generate video',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const videoId = searchParams.get('video_id')

  if (!videoId) {
    return NextResponse.json({ 
      usage: 'POST with { script, avatarId? } to generate, GET with ?video_id=X to check status',
      availableAvatars: KELLY_AVATARS
    })
  }

  // Check video status
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
    raw: data
  })
}
