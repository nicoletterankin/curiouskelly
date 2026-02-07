import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

// Adult Kelly Storyteller avatar
const KELLY_AVATAR_ID = '3d6a9d6f91b444469dae87ebb3d9eba6'

export async function GET() {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }

  // Get Day 18 hook script from database
  const lessons = await sql`
    SELECT hook_script, title FROM lessons WHERE day_of_year = 18 LIMIT 1
  `
  
  if (lessons.length === 0) {
    return NextResponse.json({ error: 'No lesson found for Day 18' }, { status: 404 })
  }

  const script = lessons[0].hook_script || 'While you sleep, your brain is busier than when you are awake.'
  const title = lessons[0].title

  // Check if we already have this video
  const existing = await sql`
    SELECT * FROM heygen_videos 
    WHERE day_of_year = 18 AND phase = 'hook' AND age_category = 'adult' AND archetype = 'storyteller'
  `

  if (existing.length > 0 && existing[0].video_url) {
    return NextResponse.json({
      status: 'already_complete',
      video_url: existing[0].video_url,
      message: 'Video already exists in database'
    })
  }

  if (existing.length > 0 && existing[0].heygen_video_id) {
    // Already submitted, check status
    const statusResponse = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${existing[0].heygen_video_id}`,
      { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
    )
    const statusData = await statusResponse.json()

    if (statusData.data?.status === 'completed') {
      // Update database with video URL
      await sql`
        UPDATE heygen_videos 
        SET video_url = ${statusData.data.video_url}, status = 'completed', completed_at = NOW()
        WHERE heygen_video_id = ${existing[0].heygen_video_id}
      `
      
      // Note: kelly_lesson_assets is for audio only, not video
      // Video URLs are stored in heygen_videos table (already updated above)

      return NextResponse.json({
        status: 'completed',
        video_url: statusData.data.video_url,
        heygen_video_id: existing[0].heygen_video_id,
        message: 'Video completed and saved!'
      })
    }

    return NextResponse.json({
      status: statusData.data?.status || 'unknown',
      heygen_video_id: existing[0].heygen_video_id,
      message: 'Video still processing, call again to check'
    })
  }

  // Submit new video to HeyGen
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: KELLY_AVATAR_ID
      },
      voice: {
        type: 'text',
        input_text: script,
        voice_id: ELEVENLABS_VOICE_ID
      }
    }],
    dimension: { width: 1080, height: 1920 }
  }

  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })

  const data = await response.json()

  if (!response.ok || data.error) {
    return NextResponse.json({ 
      error: 'HeyGen API error', 
      details: data,
      script_used: script
    }, { status: 500 })
  }

  const videoId = data.data?.video_id

  if (!videoId) {
    return NextResponse.json({ error: 'No video_id returned', data }, { status: 500 })
  }

  // Save to database
  await sql`
    INSERT INTO heygen_videos (
      day_of_year, phase, age_category, archetype, 
      heygen_video_id, script, avatar_key, status, created_at
    ) VALUES (
      18, 'hook', 'adult', 'storyteller',
      ${videoId}, ${script}, ${KELLY_AVATAR_ID}, 'processing', NOW()
    )
    ON CONFLICT (day_of_year, phase, age_category, archetype) 
    DO UPDATE SET heygen_video_id = EXCLUDED.heygen_video_id, status = 'processing', updated_at = NOW()
  `

  return NextResponse.json({
    status: 'submitted',
    heygen_video_id: videoId,
    title,
    script,
    message: 'Video submitted to HeyGen! Call this endpoint again in 2-3 minutes to check status.'
  })
}
