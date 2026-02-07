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

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { 
      visitor_id, 
      event_type, // video_start, vote, phase_change, page_view, cta_click
      event_data, // JSON with context (day, phase, vote value, etc.)
      page,
      timestamp 
    } = body

    const sql = getSql()
    await sql`
      INSERT INTO visitor_events (
        visitor_id,
        event_type,
        event_data,
        page,
        created_at
      ) VALUES (
        ${visitor_id},
        ${event_type},
        ${JSON.stringify(event_data || {})},
        ${page || '/'},
        ${timestamp || new Date().toISOString()}
      )
    `

    return NextResponse.json({ success: true })
  } catch (error) {
    // Fail silently - never break the user experience
    console.error('[visitor/event] Error:', error)
    return NextResponse.json({ success: false }, { status: 200 })
  }
}
