import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// In-memory store for real-time data (use Redis in production)
const recentEvents: Array<{
  type: string
  sessionId: string
  timestamp: string
  data: Record<string, unknown>
}> = []

const MAX_EVENTS = 1000

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    
    // Get IP and geo info from Vercel headers
    const ip = request.headers.get('x-forwarded-for')?.split(',')[0] || 'unknown'
    const anonymizedIp = ip.includes('.') 
      ? ip.substring(0, ip.lastIndexOf('.')) + '.xxx'
      : ip
    
    const geo = {
      city: request.headers.get('x-vercel-ip-city') || 'Unknown',
      region: request.headers.get('x-vercel-ip-country-region') || 'Unknown',
      country: request.headers.get('x-vercel-ip-country') || 'Unknown',
    }
    
    const userAgent = request.headers.get('user-agent') || ''
    
    const enrichedData = {
      ...data,
      ip: anonymizedIp,
      geo,
      userAgent,
      receivedAt: new Date().toISOString(),
    }
    
    // Store in memory for real-time dashboard
    recentEvents.unshift({
      type: data.type,
      sessionId: data.sessionId,
      timestamp: new Date().toISOString(),
      data: enrichedData,
    })
    
    // Keep only recent events
    if (recentEvents.length > MAX_EVENTS) {
      recentEvents.pop()
    }
    
    // Store in database for persistence
    try {
      await sql`
        INSERT INTO analytics_events (
          session_id,
          event_type,
          event_data,
          ip_address,
          city,
          region,
          country,
          user_agent,
          created_at
        ) VALUES (
          ${data.sessionId},
          ${data.type},
          ${JSON.stringify(enrichedData)}::jsonb,
          ${anonymizedIp},
          ${geo.city},
          ${geo.region},
          ${geo.country},
          ${userAgent},
          NOW()
        )
      `
    } catch (dbError) {
      // DB error is non-critical - continue silently
      console.warn('[track] DB insert failed:', dbError)
    }
    
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('[track] Error:', error)
    return NextResponse.json({ success: false }, { status: 200 }) // Return 200 to not break client
  }
}

// GET endpoint to fetch recent events (for dashboard)
export async function GET() {
  return NextResponse.json({
    events: recentEvents.slice(0, 100),
    count: recentEvents.length,
  })
}
