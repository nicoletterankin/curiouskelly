import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// Simple cache to reduce DB calls
const liveClassCache = new Map<string, { data: unknown; timestamp: number }>()
const CACHE_TTL = 15 * 1000 // 15 seconds

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const ageBracket = searchParams.get('age')
    
    // Check cache first
    const cacheKey = ageBracket || 'all'
    const cached = liveClassCache.get(cacheKey)
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      return NextResponse.json(cached.data)
    }

    // Build query based on whether age filter is provided
    let nextClass

    try {
      if (ageBracket) {
        const results = await sql`
          SELECT *
          FROM live_classes
          WHERE status IN ('scheduled', 'live')
            AND (scheduled_at > NOW() OR status = 'live')
            AND kelly_age = ${ageBracket === 'kid' ? 7 : ageBracket === 'teen' ? 14 : 35}
          ORDER BY 
            CASE WHEN status = 'live' THEN 0 ELSE 1 END,
            scheduled_at ASC
          LIMIT 1
        `
        nextClass = results[0]
      } else {
        const results = await sql`
          SELECT *
          FROM live_classes
          WHERE status IN ('scheduled', 'live')
            AND (scheduled_at > NOW() OR status = 'live')
          ORDER BY 
            CASE WHEN status = 'live' THEN 0 ELSE 1 END,
            scheduled_at ASC
          LIMIT 1
        `
        nextClass = results[0]
      }
    } catch (dbError) {
      console.warn('[live-classes] DB error, returning cached or null')
      const cachedData = liveClassCache.get(cacheKey)
      if (cachedData) {
        return NextResponse.json(cachedData.data)
      }
      return NextResponse.json(null, { status: 200 })
    }

    if (!nextClass) {
      liveClassCache.set(cacheKey, { data: null, timestamp: Date.now() })
      return NextResponse.json(null, { status: 200 })
    }

    const result = {
      id: nextClass.id,
      day: nextClass.day_of_year || 0,
      age_bracket: nextClass.kelly_age === 7 ? 'kid' : nextClass.kelly_age === 14 ? 'teen' : 'adult',
      scheduled_at: nextClass.scheduled_at,
      started_at: nextClass.started_at,
      ended_at: nextClass.ended_at,
      status: nextClass.status,
      participant_count: nextClass.peak_viewers || 0,
      max_participants: 100,
      kelly_present: nextClass.status === 'live',
      title: nextClass.title,
    }
    liveClassCache.set(cacheKey, { data: result, timestamp: Date.now() })
    return NextResponse.json(result)
  } catch (error) {
    console.error('Error fetching next live class:', error)
    // On error, return null (no live class) rather than error
    // This prevents app crashes during rate limiting
    return NextResponse.json(null, { status: 200 })
  }
}
