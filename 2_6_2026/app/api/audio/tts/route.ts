/**
 * REAL-TIME TTS GENERATION ENDPOINT
 * 
 * Generates TTS audio on-demand for a specific phase.
 * Supports: ElevenLabs (production) and Cloudflare Worker (for language/age/archetype voice matching)
 * Returns audio URL immediately if cached, otherwise generates and caches.
 * 
 * GET /api/audio/tts?day=18&phase=hook&age=adult&language=en&archetype=explorer
 * 
 * Returns: { audioUrl: string, cached: boolean, duration?: number }
 */

import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { put } from '@vercel/blob'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

// Cloudflare Worker TTS endpoint (supports language/age/archetype voice matching)
const CLOUDFLARE_TTS_URL = 'https://curious-kelly-tts.nicoletterankin.workers.dev/tts'

// Map frontend age to TTS age variant
const AGE_MAP: Record<string, string> = {
  'child': 'kid',
  'kid': 'kid',
  'teen': 'teen',
  'adult': 'adult',
  'middleAge': 'adult',
  'senior': 'elder',
  'elder': 'elder',
}

// Phase script column mapping
const SCRIPT_COLUMNS: Record<string, string> = {
  'hook': 'hook_script',
  'story': 'story_script',
  'wonder': 'wonder_script',
  'action': 'action_script',
  'wisdom': 'wisdom_script',
}

// Map language codes to supported TTS languages
const LANGUAGE_MAP: Record<string, string> = {
  'en': 'en',
  'es': 'es',
  'pt': 'pt',
  'fr': 'fr',
  'de': 'de',
  'zh': 'zh',
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '0', 10)
  const phase = searchParams.get('phase') || 'hook'
  const age = searchParams.get('age') || 'adult'
  const language = searchParams.get('language') || 'en'
  const archetype = searchParams.get('archetype') || 'explorer'

  if (!day || day < 1 || day > 365) {
    return NextResponse.json({ error: 'Invalid day' }, { status: 400 })
  }

  if (!SCRIPT_COLUMNS[phase]) {
    return NextResponse.json({ error: 'Invalid phase' }, { status: 400 })
  }

  const audioAge = AGE_MAP[age] || 'adult'
  const ttsLanguage = LANGUAGE_MAP[language] || 'en'
  const ageVariant = `${audioAge}_${phase}_${ttsLanguage}_${archetype}`

  try {
    // 1. Check if audio already exists in lesson_audio (with full variant key)
    const existing = await sql`
      SELECT audio_url, duration_seconds
      FROM lesson_audio
      WHERE day_number = ${day}
        AND language = ${ttsLanguage}
        AND age_variant = ${ageVariant}
        AND audio_url IS NOT NULL
      LIMIT 1
    `

    if (existing && existing.length > 0) {
      return NextResponse.json({
        audioUrl: existing[0].audio_url,
        cached: true,
        duration: existing[0].duration_seconds,
      })
    }

    // 2. First try lesson_perspectives for personalized content
    let script: string | null = null
    const scriptCol = SCRIPT_COLUMNS[phase]
    
    // Try lesson_perspectives table (personalized by age/archetype/language)
    try {
      const perspectives = await sql`
        SELECT content
        FROM lesson_perspectives
        WHERE day = ${day}
          AND phase = ${phase}
          AND age_group = ${audioAge}
          AND archetype = ${archetype}
          AND language = ${ttsLanguage}
        LIMIT 1
      `
      if (perspectives && perspectives.length > 0) {
        script = perspectives[0].content as string
      }
    } catch {
      // lesson_perspectives table may not exist yet - fall through to base lessons
    }

    // 3. Fall back to base lessons table if no perspective found
    if (!script) {
      // Try translated content first
      if (ttsLanguage !== 'en') {
        try {
          const translations = await sql`
            SELECT ${sql.unsafe(scriptCol)} as script
            FROM lesson_translations
            WHERE day_of_year = ${day}
              AND language_code = ${ttsLanguage}
              AND status = 'published'
            LIMIT 1
          `
          if (translations && translations.length > 0 && translations[0].script) {
            script = translations[0].script as string
          }
        } catch {
          // Translation table may not have this content
        }
      }
      
      // Fall back to base English content
      if (!script) {
        const lessons = await sql`
          SELECT ${sql.unsafe(scriptCol)} as script, title
          FROM lessons
          WHERE day_of_year = ${day}
          LIMIT 1
        `

        if (!lessons || lessons.length === 0) {
          return NextResponse.json({ error: 'Lesson not found' }, { status: 404 })
        }

        script = lessons[0].script as string
      }
    }

    if (!script) {
      return NextResponse.json({ error: 'No script for this phase' }, { status: 404 })
    }

    // 4. Try Cloudflare Worker TTS first (supports language/age/archetype voice matching)
    try {
      const cfResponse = await fetch(CLOUDFLARE_TTS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: script,
          language: ttsLanguage,
          age: audioAge,
          archetype: archetype,
        }),
      })

      if (cfResponse.ok) {
        const cfData = await cfResponse.json()
        if (cfData.audioUrl) {
          // Cache the Cloudflare-generated audio
          await sql`
            INSERT INTO lesson_audio (
              id, day_number, language, age_variant, audio_url, character_count, created_at
            ) VALUES (
              gen_random_uuid(),
              ${day},
              ${ttsLanguage},
              ${ageVariant},
              ${cfData.audioUrl},
              ${script.length},
              NOW()
            )
            ON CONFLICT (day_number, language, age_variant) DO UPDATE SET
              audio_url = ${cfData.audioUrl},
              character_count = ${script.length}
          `.catch(() => {}) // Ignore cache errors

          return NextResponse.json({
            audioUrl: cfData.audioUrl,
            cached: false,
            generated: true,
            source: 'cloudflare',
            language: ttsLanguage,
            age: audioAge,
            archetype: archetype,
          })
        }
      }
    } catch (cfError) {
      console.warn('[tts] Cloudflare Worker unavailable, falling back to ElevenLabs:', cfError)
    }

    // 5. Fall back to ElevenLabs (multilingual but no age/archetype voice matching)
    const ttsResponse = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': ELEVENLABS_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: script,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
          },
        }),
      }
    )

    if (!ttsResponse.ok) {
      const errorText = await ttsResponse.text()
      console.error('[tts] ElevenLabs error:', ttsResponse.status, errorText)
      
      // Handle rate limiting gracefully
      if (ttsResponse.status === 429) {
        return NextResponse.json({ 
          error: 'Rate limited - try again later',
          rateLimited: true 
        }, { status: 429 })
      }
      
      // Handle quota exceeded (401 with quota_exceeded)
      if (ttsResponse.status === 401 && errorText.includes('quota_exceeded')) {
        console.warn('[tts] ElevenLabs quota exceeded - returning no audio')
        return NextResponse.json({ 
          audioUrl: null,
          cached: false,
          quotaExceeded: true,
          message: 'ElevenLabs credits exhausted - audio unavailable'
        }, { status: 200 }) // Return 200 so frontend handles gracefully
      }
      
      return NextResponse.json({ 
        error: 'TTS generation failed',
        status: ttsResponse.status 
      }, { status: 500 })
    }

    // 4. Get audio buffer
    const audioBuffer = await ttsResponse.arrayBuffer()

    // 5. Upload to Vercel Blob
    const filename = `audio/day${day.toString().padStart(3, '0')}/${phase}_${audioAge}_${language}.mp3`
    const blob = await put(filename, Buffer.from(audioBuffer), {
      access: 'public',
      contentType: 'audio/mpeg',
    })

    // 6. Store in lesson_audio table
    await sql`
      INSERT INTO lesson_audio (
        id, day_number, language, age_variant, audio_url, character_count, created_at
      ) VALUES (
        gen_random_uuid(),
        ${day},
        ${language},
        ${ageVariant},
        ${blob.url},
        ${script.length},
        NOW()
      )
      ON CONFLICT (day_number, language, age_variant) DO UPDATE SET
        audio_url = ${blob.url},
        character_count = ${script.length}
    `

    return NextResponse.json({
      audioUrl: blob.url,
      cached: false,
      generated: true,
      characterCount: script.length,
    })

  } catch (error) {
    console.error('[tts] Error:', error)
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}
