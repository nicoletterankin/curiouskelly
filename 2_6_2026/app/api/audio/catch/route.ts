/**
 * ELEVENLABS AUDIO CATCH ENDPOINT
 * 
 * Receives audio from Antigravity and stores in database.
 * 
 * POST /api/audio/catch
 * Body: {
 *   day_number: 18,
 *   phase: "hook",
 *   age_group: "adult", 
 *   archetype: "storyteller",
 *   language: "en",
 *   audio_url: "https://...",
 *   script: "...",
 *   duration_seconds: 45.2,
 *   elevenlabs_history_id: "..." (optional)
 * }
 * 
 * Or batch:
 * Body: { items: [...] }
 */

import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

export async function GET() {
  // Show pending audio that needs catching
  const sql = getSql()
  const pending = await sql`
    SELECT day_number, phase, age_group, archetype, status, audio_url
    FROM kelly_audio
    WHERE status IN ('pending', 'processing')
    ORDER BY day_number, phase
    LIMIT 50
  `.catch(() => [])
  
  const completed = await sql`
    SELECT COUNT(*)::int as count FROM kelly_audio WHERE status = 'completed'
  `.catch(() => [{ count: 0 }])
  
  return Response.json({
    status: 'ready_to_catch',
    voice_id: ELEVENLABS_VOICE_ID,
    pending: pending.length,
    completed: completed[0]?.count || 0,
    pending_samples: pending.slice(0, 10)
  })
}

export async function POST(request: Request) {
  const body = await request.json()
  
  // Handle batch or single
  const items = body.items || [body]
  const sql = getSql()
  
  const results: { day: number; phase: string; age: string; archetype: string; status: string; error?: string }[] = []
  
  for (const item of items) {
    const {
      day_number,
      phase,
      age_group,
      archetype,
      language = 'en',
      audio_url,
      script,
      duration_seconds,
      elevenlabs_history_id
    } = item
    
    // Validate required fields
    if (!day_number || !phase || !age_group || !archetype || !audio_url) {
      results.push({
        day: day_number,
        phase,
        age: age_group,
        archetype,
        status: 'failed',
        error: 'Missing required fields: day_number, phase, age_group, archetype, audio_url'
      })
      continue
    }
    
    try {
      // Upsert into kelly_audio
      await sql`
        INSERT INTO kelly_audio (
          day_number, phase, age_group, archetype, language,
          script, audio_url, duration_seconds, 
          elevenlabs_voice_id, elevenlabs_history_id,
          status, completed_at
        )
        VALUES (
          ${day_number}, ${phase}, ${age_group}, ${archetype}, ${language},
          ${script || ''}, ${audio_url}, ${duration_seconds || null},
          ${ELEVENLABS_VOICE_ID}, ${elevenlabs_history_id || null},
          'completed', NOW()
        )
        ON CONFLICT (day_number, phase, age_group, archetype, language) 
        DO UPDATE SET
          audio_url = ${audio_url},
          duration_seconds = ${duration_seconds || null},
          elevenlabs_history_id = ${elevenlabs_history_id || null},
          status = 'completed',
          completed_at = NOW()
      `
      
      // Also update kelly_lesson_assets if it exists
      await sql`
        UPDATE kelly_lesson_assets
        SET audio_url = ${audio_url}
        WHERE day_number = ${day_number}
          AND phase = ${phase}
          AND age_group = ${age_group}
          AND archetype = ${archetype}
      `.catch(() => {})
      
      results.push({
        day: day_number,
        phase,
        age: age_group,
        archetype,
        status: 'caught'
      })
      
    } catch (error) {
      results.push({
        day: day_number,
        phase,
        age: age_group,
        archetype,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }
  
  return Response.json({
    caught: results.filter(r => r.status === 'caught').length,
    failed: results.filter(r => r.status === 'failed').length,
    results
  })
}
