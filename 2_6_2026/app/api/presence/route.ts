import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// Cache to prevent rate limiting - presence data doesn't need real-time accuracy
const cache: { count: number; realUsers: number; timestamp: number } = {
  count: 15,
  realUsers: 0,
  timestamp: 0
}
const CACHE_TTL = 120000 // 2 minute cache - presence is not critical

export async function GET() {
  // ALWAYS return cached data if we have any - avoid DB calls when possible
  const now = Date.now()
  const cacheAge = now - cache.timestamp
  
  // Return cached data with slight variation
  if (cacheAge < CACHE_TTL || cache.timestamp > 0) {
    const variance = Math.floor(Math.random() * 3) - 1
    const count = Math.max(cache.count + variance, 10)
    
    // Only try DB refresh if cache is stale AND we haven't tried recently
    if (cacheAge >= CACHE_TTL) {
      // Fire and forget - don't wait for DB
      refreshCacheInBackground()
    }
    
    return NextResponse.json({ 
      count,
      realUsers: cache.realUsers,
      cached: true
    })
  }
  
  // First load - no cache yet, must query DB
  try {
    const result = await sql`
      SELECT COUNT(DISTINCT user_id) as count
      FROM presence
      WHERE last_seen > NOW() - INTERVAL '5 minutes'
    `
    
    const dbCount = Number(result[0]?.count || 0)
    const count = Math.max(dbCount + 8 + Math.floor(Math.random() * 8), 12)
    
    cache.count = count
    cache.realUsers = dbCount
    cache.timestamp = now
    
    return NextResponse.json({ count, realUsers: dbCount })
  } catch {
    // DB error - return fallback
    cache.timestamp = now // Mark as "tried" to prevent retry storm
    return NextResponse.json({ count: 15, realUsers: 0, fallback: true })
  }
}

// Background cache refresh - non-blocking
async function refreshCacheInBackground() {
  try {
    const result = await sql`
      SELECT COUNT(DISTINCT user_id) as count
      FROM presence
      WHERE last_seen > NOW() - INTERVAL '5 minutes'
    `
    const dbCount = Number(result[0]?.count || 0)
    cache.count = Math.max(dbCount + 8 + Math.floor(Math.random() * 8), 12)
    cache.realUsers = dbCount
    cache.timestamp = Date.now()
  } catch {
    // Silent fail - keep using stale cache
  }
}

export async function POST(request: NextRequest) {
  try {
    const { fingerprint } = await request.json()
    
    if (!fingerprint) {
      return NextResponse.json({ error: 'Missing fingerprint' }, { status: 400 })
    }
    
    // Simple update/insert without complex JSONB - just track user presence
    try {
      // Try update first
      const updated = await sql`
        UPDATE presence 
        SET last_seen = NOW()
        WHERE user_id = ${fingerprint}::text
        RETURNING id
      `
      
      // Insert if no existing record
      if (updated.length === 0) {
        await sql`
          INSERT INTO presence (id, user_id, last_seen)
          VALUES (gen_random_uuid(), ${fingerprint}::text, NOW())
        `
      }
    } catch {
      // Ignore DB errors for presence - it's not critical
    }
    
    // Get updated count
    const result = await sql`
      SELECT COUNT(DISTINCT user_id) as count
      FROM presence
      WHERE last_seen > NOW() - INTERVAL '5 minutes'
    `.catch(() => [{ count: 0 }])
    
    const dbCount = Number(result[0]?.count || 0)
    const count = Math.max(dbCount + 8 + Math.floor(Math.random() * 8), 12)
    
    return NextResponse.json({ count, realUsers: dbCount })
  } catch (error) {
    console.error('Presence POST error:', error)
    return NextResponse.json({ count: 14, realUsers: 0 })
  }
}
