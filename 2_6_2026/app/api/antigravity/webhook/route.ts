import { NextRequest, NextResponse } from 'next/server'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

/**
 * ANTIGRAVITY WEBHOOK
 * 
 * Antigravity calls this endpoint when assets are ready.
 * No complex protocol - just POST JSON.
 * 
 * POST /api/antigravity/webhook
 * {
 *   "action": "asset_complete",
 *   "type": "audio" | "video",
 *   "day": 19,
 *   "phase": "hook" | "story" | "wonder" | "action" | "wisdom",
 *   "language": "en" | "es" | "fr" | "de" | "pt" | "zh",
 *   "age": 8 | 14 | 35,
 *   "url": "https://..."
 * }
 * 
 * Response: { "ok": true } or { "error": "..." }
 */

export async function POST(request: NextRequest) {
  try {
    // Simple auth - check for secret header
    const secret = request.headers.get('x-antigravity-secret')
    if (secret !== process.env.SUPABASE_WEBHOOK_SECRET) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const body = await request.json()
    const { action, type, day, phase, language, age, url } = body

    if (action === 'asset_complete') {
      // Validate required fields
      if (!type || !day || !phase || !url) {
        return NextResponse.json({ 
          error: 'Missing fields. Required: type, day, phase, url' 
        }, { status: 400 })
      }

      // generated_assets schema: id, lesson_id (integer), day_number, phase, asset_type, status, url, prompt, created_at
      // Upsert into generated_assets - use day_number as integer
      const sql = getSql()
      await sql`
        INSERT INTO generated_assets (id, lesson_id, day_number, phase, asset_type, status, url, created_at)
        VALUES (gen_random_uuid(), ${day}, ${day}, ${phase}, ${type}, 'completed', ${url}, NOW())
        ON CONFLICT (id) 
        DO UPDATE SET url = ${url}, status = 'completed'
      `

      return NextResponse.json({ 
        ok: true, 
        message: `${type}/day-${day}/${phase} registered` 
      })
    }

    if (action === 'status') {
      // Return what's missing for a given day
      const targetDay = day || 19
      
      const sql = getSql()
      const existing = await sql`
        SELECT DISTINCT phase, asset_type 
        FROM generated_assets 
        WHERE day_number = ${targetDay}
        AND status = 'completed'
      `

      const phases = ['hook', 'story', 'wonder', 'action', 'wisdom']
      const types = ['audio', 'video']

      const existingSet = new Set(
        existing.map((r: { phase: string; asset_type: string }) => 
          `${r.asset_type}/${r.phase}`
        )
      )

      const missing: string[] = []
      const complete: string[] = []

      for (const t of types) {
        for (const p of phases) {
          const key = `${t}/${p}`
          if (existingSet.has(key)) {
            complete.push(key)
          } else {
            missing.push(key)
          }
        }
      }

      return NextResponse.json({
        day: targetDay,
        total_needed: phases.length * types.length,
        complete: complete.length,
        missing: missing.length,
        missing_list: missing,
        complete_list: complete
      })
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 })

  } catch (error) {
    console.error('Antigravity webhook error:', error)
    return NextResponse.json({ error: 'Internal error' }, { status: 500 })
  }
}

// GET - Quick status check (no auth required)
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '19')

  try {
    const sql = getSql()
    const counts = await sql`
      SELECT asset_type, COUNT(DISTINCT phase) as phases
      FROM generated_assets 
      WHERE day_number = ${day}
      AND status = 'completed'
      GROUP BY asset_type
    `

    // Format: { audio: 5, video: 0 }
    const status: Record<string, number> = { audio: 0, video: 0 }

    for (const row of counts) {
      const r = row as { asset_type: string; phases: number }
      status[r.asset_type] = Number(r.phases)
    }

    return NextResponse.json({ day, status })
  } catch {
    return NextResponse.json({ day, status: {}, error: 'DB error' })
  }
}
