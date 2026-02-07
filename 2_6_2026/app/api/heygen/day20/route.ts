import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Kelly Adult Storyteller - talking photo ID from working test-one route
const KELLY_AVATAR_ID = process.env.KELLY_AVATAR_ID || '3d6a9d6f91b444469dae87ebb3d9eba6'

// Day 20: January 20, 2026 - "Why We Yawn"
const DAY = 20
const PHASE = 'hook'
const AGE_CATEGORY = 'adult'
const ARCHETYPE = 'storyteller'

const SCRIPT = `Have you noticed that yawns are contagious? When someone yawns, you might suddenly feel the urge to yawn too! Scientists think this happens because of mirror neurons in our brain that help us connect with others. But here's the really curious part - yawning isn't just about being tired. Your brain actually uses yawning to cool itself down, like a built-in air conditioner! Today we're going to explore why we yawn and what it tells us about how our amazing brains work.`

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  const videoId = searchParams.get('video_id')

  // Check if API key is configured
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({
      error: 'HEYGEN_API_KEY not configured',
      fix: 'Add HEYGEN_API_KEY to environment variables in Vercel dashboard',
      alternative: 'Use /api/fal/day20 for fal.ai video generation instead'
    }, { status: 500 })
  }

  // Check status of existing video
  if (action === 'status' && videoId) {
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

  // Generate the video
  if (action === 'generate') {
    console.log('[HeyGen Day20] Generating video...')
    console.log('[HeyGen Day20] Avatar ID:', KELLY_AVATAR_ID)
    console.log('[HeyGen Day20] Script:', SCRIPT)

    // EXACT PAYLOAD - Using talking_photo type (confirmed working)
    const payload = {
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: KELLY_AVATAR_ID
        },
        voice: {
          type: 'text',
          input_text: SCRIPT,
          voice_id: '1bd001e7e50f421d891986aad5158bc8' // HeyGen female voice
        }
      }],
      dimension: {
        width: 1920,  // Apple Keynote quality
        height: 1080
      },
      test: false
    }

    console.log('[HeyGen Day20] EXACT PAYLOAD:', JSON.stringify(payload, null, 2))

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
        error: 'HeyGen API error',
        status: res.status,
        response: data,
        payload_sent: payload,
        troubleshooting: {
          avatar_id_used: KELLY_AVATAR_ID,
          api_key_set: !!HEYGEN_API_KEY,
          api_key_prefix: HEYGEN_API_KEY?.substring(0, 8) + '...'
        }
      }, { status: res.status })
    }

    const newVideoId = data.data?.video_id

    // Store in database
    try {
      const sql = getSql()
      await sql`
        INSERT INTO heygen_videos (
          day_of_year, phase, age_category, archetype,
          heygen_video_id, script, avatar_id, status, created_at
        ) VALUES (
          ${DAY}, ${PHASE}, ${AGE_CATEGORY}, ${ARCHETYPE},
          ${newVideoId}, ${SCRIPT}, ${KELLY_AVATAR_ID}, 'processing', NOW()
        )
        ON CONFLICT (day_of_year, phase, age_category, archetype)
        DO UPDATE SET 
          heygen_video_id = ${newVideoId},
          script = ${SCRIPT},
          status = 'processing',
          created_at = NOW()
      `
    } catch (dbErr) {
      console.warn('[HeyGen Day20] DB insert failed:', dbErr)
    }

    return NextResponse.json({
      success: true,
      videoId: newVideoId,
      status: 'processing',
      check_status: `/api/heygen/day20?action=status&video_id=${newVideoId}`,
      message: 'Video generation started! Check status in 30-60 seconds.',
      payload_sent: payload,
      config: {
        day: DAY,
        phase: PHASE,
        age: AGE_CATEGORY,
        archetype: ARCHETYPE,
        avatar_id: KELLY_AVATAR_ID,
        script_length: SCRIPT.length
      }
    })
  }

  // Default: show instructions
  return NextResponse.json({
    endpoint: '/api/heygen/day20',
    day: DAY,
    title: 'Why We Yawn',
    date: 'January 20, 2026',
    actions: {
      generate: '/api/heygen/day20?action=generate',
      status: '/api/heygen/day20?action=status&video_id=VIDEO_ID'
    },
    config: {
      avatar_id: KELLY_AVATAR_ID,
      age_category: AGE_CATEGORY,
      archetype: ARCHETYPE,
      phase: PHASE,
      script_preview: SCRIPT.substring(0, 100) + '...'
    }
  })
}
