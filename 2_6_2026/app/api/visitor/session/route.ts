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
      referrer, 
      landing_page, 
      user_agent, 
      timestamp,
      intent // Optional: investor, learner, guardian, builder
    } = body

    // Parse user agent for device type
    const isMobile = /Mobile|Android|iPhone|iPad/i.test(user_agent || '')
    const isTablet = /iPad|Tablet/i.test(user_agent || '')
    const deviceType = isTablet ? 'tablet' : isMobile ? 'mobile' : 'desktop'

    // Infer intent from landing page and referrer
    let inferredIntent = intent || 'learner'
    if (landing_page?.includes('/strategic') || landing_page?.includes('/investors')) {
      inferredIntent = 'investor'
    } else if (referrer?.includes('linkedin') || referrer?.includes('investor')) {
      inferredIntent = 'investor'
    } else if (landing_page?.includes('/about') || landing_page?.includes('/command')) {
      inferredIntent = 'builder'
    }

    // Insert or update visitor session
    const sql = getSql()
    await sql`
      INSERT INTO visitor_sessions (
        visitor_id,
        referrer,
        landing_page,
        user_agent,
        device_type,
        intent,
        first_seen,
        last_seen,
        session_count
      ) VALUES (
        ${visitor_id},
        ${referrer || null},
        ${landing_page || '/'},
        ${user_agent || null},
        ${deviceType},
        ${inferredIntent},
        ${timestamp || new Date().toISOString()},
        ${timestamp || new Date().toISOString()},
        1
      )
      ON CONFLICT (visitor_id) DO UPDATE SET
        last_seen = ${timestamp || new Date().toISOString()},
        session_count = visitor_sessions.session_count + 1,
        intent = COALESCE(${inferredIntent}, visitor_sessions.intent)
    `

    return NextResponse.json({ success: true, intent: inferredIntent })
  } catch (error) {
    // Fail silently - never break the user experience
    console.error('[visitor/session] Error:', error)
    return NextResponse.json({ success: false }, { status: 200 })
  }
}
