/**
 * EXECUTE FULL AUDIO GENERATION FOR DAY 34
 * 
 * This endpoint actually generates audio for all day 34 assets.
 * POST /api/admin/generate-day-34
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

// Cloudflare TTS
const CLOUDFLARE_TTS_URL = 'https://curious-kelly-tts.nicoletterankin.workers.dev/tts'

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  const results: Array<{id: string, phase: string, status: string, audioUrl?: string, error?: string}> = []
  
  try {
    const sql = getSql()
    // Get all day 34 assets needing audio (English only for now to be faster)
    const assets = await sql`
      SELECT id, phase, age_group, language, archetype, script_text 
      FROM kelly_lesson_assets 
      WHERE day_number = 34 
        AND audio_url IS NULL 
        AND language = 'en'
      ORDER BY phase, age_group
      LIMIT 50
    `

    if (!assets || assets.length === 0) {
      return NextResponse.json({ 
        message: 'No assets need audio generation',
        alreadyComplete: true 
      })
    }

    console.log(`[generate-day-34] Starting generation for ${assets.length} assets`)

    // Process each asset
    for (const asset of assets) {
      const { id, phase, age_group, language, archetype, script_text } = asset as {
        id: string
        phase: string
        age_group: string
        language: string
        archetype: string | null
        script_text: string
      }

      try {
        // Try Cloudflare TTS first
        let audioUrl: string | null = null
        
        try {
          const cfResponse = await fetch(CLOUDFLARE_TTS_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: script_text,
              language: language || 'en',
              age: age_group || 'adult',
              archetype: archetype || 'explorer',
            }),
          })

          if (cfResponse.ok) {
            const cfData = await cfResponse.json()
            if (cfData.audioUrl) {
              audioUrl = cfData.audioUrl
            }
          }
        } catch (cfError) {
          console.warn(`[generate-day-34] Cloudflare TTS failed for ${id}:`, cfError)
        }

        // Fallback to ElevenLabs
        if (!audioUrl && ELEVENLABS_API_KEY) {
          const ttsResponse = await fetch(
            `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
            {
              method: 'POST',
              headers: {
                'xi-api-key': ELEVENLABS_API_KEY,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                text: script_text,
                model_id: 'eleven_multilingual_v2',
                voice_settings: {
                  stability: 0.5,
                  similarity_boost: 0.75,
                },
              }),
            }
          )

          if (ttsResponse.ok) {
            const audioBuffer = await ttsResponse.arrayBuffer()
            const filename = `audio/day034/${phase}_${age_group}_${language}.mp3`
            const blob = await put(filename, Buffer.from(audioBuffer), {
              access: 'public',
              contentType: 'audio/mpeg',
            })
            audioUrl = blob.url
          }
        }

        if (audioUrl) {
          // Update the asset with the audio URL
          await sql`
            UPDATE kelly_lesson_assets 
            SET audio_url = ${audioUrl}, 
                status = 'audio_ready',
                updated_at = NOW()
            WHERE id = ${id}::uuid
          `
          
          results.push({ id, phase, status: 'success', audioUrl })
          console.log(`[generate-day-34] Generated audio for ${phase}/${age_group}`)
        } else {
          results.push({ id, phase, status: 'no_audio', error: 'No TTS provider available' })
        }

        // Rate limiting - wait between requests
        await new Promise(r => setTimeout(r, 500))

      } catch (assetError) {
        console.error(`[generate-day-34] Error for asset ${id}:`, assetError)
        results.push({ 
          id, 
          phase, 
          status: 'error', 
          error: assetError instanceof Error ? assetError.message : 'Unknown error' 
        })
      }
    }

    const elapsed = Date.now() - startTime
    const successful = results.filter(r => r.status === 'success').length

    return NextResponse.json({
      success: true,
      day: 34,
      processed: results.length,
      successful,
      failed: results.length - successful,
      elapsedMs: elapsed,
      results,
    })

  } catch (error) {
    console.error('[generate-day-34] Fatal error:', error)
    return NextResponse.json({
      error: 'Generation failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      results,
    }, { status: 500 })
  }
}

export async function GET() {
  const sql = getSql()
  // Check current status
  const status = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video
    FROM kelly_lesson_assets 
    WHERE day_number = 34
  `

  const lesson = await sql`
    SELECT title, hook_script FROM lessons WHERE day_of_year = 34
  `

  return NextResponse.json({
    day: 34,
    lesson: lesson[0],
    assets: status[0],
    ready: (status[0] as {with_audio: number}).with_audio > 0,
  })
}
