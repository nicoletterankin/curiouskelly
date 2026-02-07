/**
 * ELEVENLABS AUDIO GENERATION ENDPOINT
 * 
 * Generates Kelly voice audio for lesson scripts.
 * Audio is generated FIRST, then used by any video engine.
 * 
 * POST /api/audio/generate
 * Body: { day: 18, phase: "hook", age: "adult", archetype: "storyteller" }
 * 
 * Or batch:
 * Body: { day: 18, batch: 10 }
 */

import { sql } from '@/lib/db'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

const AGES = ['baby', 'kid', 'teen', 'adult', 'elder']
const ARCHETYPES = ['storyteller', 'explorer', 'scientist', 'philosopher', 'coach', 'mystic']
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom']

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const day = searchParams.get('day')
  
  // Get status of audio generation from lesson_audio table
  // Schema: id, day_number, language, age_variant, audio_url, duration_seconds, character_count, created_at
  let stats
  if (day) {
    stats = await sql`
      SELECT 
        CASE WHEN audio_url IS NOT NULL THEN 'completed' ELSE 'pending' END as status,
        COUNT(*)::int as count
      FROM lesson_audio
      WHERE day_number = ${parseInt(day)}
      GROUP BY CASE WHEN audio_url IS NOT NULL THEN 'completed' ELSE 'pending' END
    `
  } else {
    stats = await sql`
      SELECT 
        CASE WHEN audio_url IS NOT NULL THEN 'completed' ELSE 'pending' END as status,
        COUNT(*)::int as count
      FROM lesson_audio
      GROUP BY CASE WHEN audio_url IS NOT NULL THEN 'completed' ELSE 'pending' END
    `
  }
  
  const recent = await sql`
    SELECT day_number, language, age_variant, audio_url, duration_seconds
    FROM lesson_audio
    ORDER BY created_at DESC
    LIMIT 20
  `
  
  return Response.json({
    status: 'ok',
    stats,
    recent_audio: recent,
    total_combinations_per_day: AGES.length * PHASES.length // 25 (age variants x phases)
  })
}

export async function POST(request: Request) {
  const body = await request.json()
  const { day, phase, age, archetype, batch = 1 } = body
  
  if (!day) {
    return Response.json({ error: 'day is required' }, { status: 400 })
  }
  
  // Get lesson script from database
  const lessons = await sql`
    SELECT day_of_year, title, hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons
    WHERE day_of_year = ${day}
  `
  
  if (lessons.length === 0) {
    return Response.json({ error: `No lesson found for day ${day}` }, { status: 404 })
  }
  
  const lesson = lessons[0]
  
  // Build list of combinations to generate
  // Using lesson_audio schema: day_number, language, age_variant, audio_url
  const combinations: { phase: string; age: string; language: string }[] = []
  const language = 'en' // Default to English
  
  if (phase && age) {
    // Single specific combination
    combinations.push({ phase, age, language })
  } else {
    // Generate batch of missing combinations
    const existing = await sql`
      SELECT age_variant
      FROM lesson_audio 
      WHERE day_number = ${day} AND audio_url IS NOT NULL
    `
    const existingSet = new Set(existing.map((e: { age_variant: string }) => e.age_variant))
    
    for (const p of phase ? [phase] : PHASES) {
      for (const a of age ? [age] : AGES) {
        const key = `${a}_${p}`
        if (!existingSet.has(key)) {
          combinations.push({ phase: p, age: a, language })
          if (combinations.length >= batch) break
        }
      }
      if (combinations.length >= batch) break
    }
  }
  
  if (combinations.length === 0) {
    return Response.json({ 
      message: 'All combinations already exist for this day',
      day 
    })
  }
  
  const results: { phase: string; age: string; language: string; status: string; audio_url?: string; error?: string }[] = []
  
  for (const combo of combinations) {
    try {
      // Get script for this phase
      const scriptKey = `${combo.phase}_script` as keyof typeof lesson
      const script = lesson[scriptKey] as string
      
      if (!script) {
        results.push({ ...combo, status: 'skipped', error: `No script for phase ${combo.phase}` })
        continue
      }
      
      const ageVariant = `${combo.age}_${combo.phase}`
      
      // Call ElevenLabs API
      const response = await fetch(
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
      
      if (!response.ok) {
        const error = await response.text()
        results.push({ ...combo, status: 'failed', error })
        continue
      }
      
      // Get audio data
      const audioBuffer = await response.arrayBuffer()
      
      // Upload to Vercel Blob
      const { put } = await import('@vercel/blob')
      const filename = `audio/day${day.toString().padStart(3, '0')}/${combo.phase}_${combo.age}.mp3`
      
      const blob = await put(filename, Buffer.from(audioBuffer), {
        access: 'public',
        contentType: 'audio/mpeg',
      })
      
      // Insert into lesson_audio table (actual schema)
      await sql`
        INSERT INTO lesson_audio (
          id, day_number, language, age_variant, audio_url, character_count, created_at
        ) VALUES (
          gen_random_uuid(),
          ${day},
          ${combo.language},
          ${ageVariant},
          ${blob.url},
          ${script.length},
          NOW()
        )
        ON CONFLICT (day_number, language, age_variant) DO UPDATE SET
          audio_url = ${blob.url},
          character_count = ${script.length}
      `
      
      results.push({ ...combo, status: 'completed', audio_url: blob.url })
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      results.push({ ...combo, status: 'failed', error: errorMsg })
    }
  }
  
  return Response.json({
    day,
    generated: results.filter(r => r.status === 'completed').length,
    failed: results.filter(r => r.status === 'failed').length,
    skipped: results.filter(r => r.status === 'skipped').length,
    results,
  })
}
