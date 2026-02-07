import { neon, NeonQueryFunction } from "@neondatabase/serverless"
import { NextResponse } from "next/server"

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

// ElevenLabs voice IDs for different languages
const VOICE_MAP: Record<string, { voiceId: string; modelId: string }> = {
  en: { voiceId: "EXAVITQu4vr4xnSDxMaL", modelId: "eleven_turbo_v2_5" }, // Sarah
  es: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" }, // Charlotte (multilingual)
  hi: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  zh: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  pt: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  ar: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  fr: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  ru: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  ja: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  de: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  ko: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
  it: { voiceId: "XB0fDUnXU5powFXDhCwa", modelId: "eleven_multilingual_v2" },
}

export async function POST(request: Request) {
  try {
    const { language, dayStart = 1, dayEnd = 7, batchSize = 10 } = await request.json()

    if (!language || !VOICE_MAP[language]) {
      return NextResponse.json({ error: "Invalid language" }, { status: 400 })
    }

    const sql = getSql()
    // Get scripts that need audio generation
    const scripts = await sql`
      SELECT id, day_number, phase, age_group, script_text
      FROM kelly_lesson_assets
      WHERE language = ${language}
        AND audio_url IS NULL
        AND script_text IS NOT NULL
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
      ORDER BY day_number, phase
      LIMIT ${batchSize}
    `

    if (scripts.length === 0) {
      return NextResponse.json({ 
        message: "No scripts need audio generation for this language/range",
        language,
        dayStart,
        dayEnd
      })
    }

    const voice = VOICE_MAP[language]
    const results = []
    const errors = []

    for (const script of scripts) {
      try {
        // Call ElevenLabs API
        const response = await fetch(
          `https://api.elevenlabs.io/v1/text-to-speech/${voice.voiceId}`,
          {
            method: "POST",
            headers: {
              "xi-api-key": process.env.ELEVENLABS_API_KEY!,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              text: script.script_text,
              model_id: voice.modelId,
              voice_settings: {
                stability: 0.5,
                similarity_boost: 0.75,
                style: 0.5,
                use_speaker_boost: true,
              },
            }),
          }
        )

        if (!response.ok) {
          const error = await response.text()
          errors.push({ id: script.id, error })
          continue
        }

        const audioBuffer = await response.arrayBuffer()
        
        // Upload to Cloudflare R2 or Vercel Blob
        const audioUrl = await uploadAudio(
          audioBuffer,
          `${language}/day-${script.day_number}-${script.phase}-${script.age_group}.mp3`
        )

        // Update database with audio URL
        await sql`
          UPDATE kelly_lesson_assets
          SET audio_url = ${audioUrl}, updated_at = NOW()
          WHERE id = ${script.id}
        `

        results.push({
          id: script.id,
          day: script.day_number,
          phase: script.phase,
          ageGroup: script.age_group,
          audioUrl,
        })
      } catch (err) {
        errors.push({ id: script.id, error: String(err) })
      }
    }

    return NextResponse.json({
      success: true,
      language,
      generated: results.length,
      errors: errors.length,
      results,
      errors: errors.length > 0 ? errors : undefined,
    })
  } catch (error) {
    console.error("Batch audio generation error:", error)
    return NextResponse.json(
      { error: "Failed to generate audio" },
      { status: 500 }
    )
  }
}

async function uploadAudio(buffer: ArrayBuffer, key: string): Promise<string> {
  // For now, use Vercel Blob if available, otherwise return placeholder
  if (process.env.BLOB_READ_WRITE_TOKEN) {
    const { put } = await import("@vercel/blob")
    const blob = await put(`audio/${key}`, Buffer.from(buffer), {
      access: "public",
      contentType: "audio/mpeg",
    })
    return blob.url
  }
  
  // Fallback: Return a data URL (not recommended for production)
  return `data:audio/mpeg;base64,${Buffer.from(buffer).toString("base64")}`
}

// GET: Check generation status for a language
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const language = searchParams.get("language") || "es"

  const sql = getSql()
  const stats = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(*) FILTER (WHERE audio_url IS NULL AND script_text IS NOT NULL) as pending
    FROM kelly_lesson_assets
    WHERE language = ${language}
  `

  return NextResponse.json({
    language,
    total: stats[0].total,
    withAudio: stats[0].with_audio,
    pending: stats[0].pending,
    percentComplete: Math.round((stats[0].with_audio / stats[0].total) * 100),
  })
}
