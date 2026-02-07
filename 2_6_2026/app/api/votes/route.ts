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

// Generate deterministic UUID from day+phase string
function generateDeterministicUUID(input: string): string {
  let hash = 0
  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash
  }
  const hex = Math.abs(hash).toString(16).padStart(8, '0')
  return `${hex.slice(0,8)}-${hex.slice(0,4)}-4${hex.slice(1,4)}-8${hex.slice(0,3)}-${hex.padStart(12, '0').slice(0,12)}`
}

// Simple object cache (avoids Map issues in Edge runtime)
const cache: Record<string, { 
  vote: string | null
  trueCount: number
  falseCount: number
  timestamp: number 
}> = {}

// Global default counts
let globalTrue = 72
let globalFalse = 28

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '21')
  const phase = searchParams.get('phase') || 'hook'
  const fingerprint = searchParams.get('fingerprint') || ''
  
  const cacheKey = `${day}-${phase}-${fingerprint}`
  const cached = cache[cacheKey]
  const now = Date.now()
  
  // Return cached data if fresh (5 minutes)
  if (cached && (now - cached.timestamp) < 300000) {
    return NextResponse.json({ 
      votes: { true: cached.trueCount, false: cached.falseCount },
      userVote: cached.vote ? { vote: cached.vote } : null,
      cached: true
    })
  }
  
  // Return global defaults
  return NextResponse.json({ 
    votes: { true: globalTrue, false: globalFalse },
    userVote: null,
    cached: true
  })
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { day, phase, fingerprint, vote } = body
    
    if (!fingerprint || !vote) {
      return NextResponse.json({ error: 'Missing fingerprint or vote' }, { status: 400 })
    }
    
    if (vote !== 'true' && vote !== 'false') {
      return NextResponse.json({ error: 'Vote must be "true" or "false"' }, { status: 400 })
    }
    
    const dayNum = parseInt(day) || 21
    const phaseStr = phase || 'hook'
    const voteValue = vote === 'true' ? 1 : -1
    const itemId = generateDeterministicUUID(`day-${dayNum}-${phaseStr}`)
    
    // Update cache immediately
    const cacheKey = `${dayNum}-${phaseStr}-${fingerprint}`
    const existing = cache[cacheKey]
    
    let trueCount = existing?.trueCount ?? globalTrue
    let falseCount = existing?.falseCount ?? globalFalse
    
    if (existing && existing.vote && existing.vote !== vote) {
      // Changing vote
      if (existing.vote === 'true') { trueCount--; falseCount++ }
      else { trueCount++; falseCount-- }
    } else if (!existing || !existing.vote) {
      // New vote
      if (vote === 'true') trueCount++
      else falseCount++
    }
    
    // Update cache using object assignment (not Map.set)
    cache[cacheKey] = { vote, trueCount, falseCount, timestamp: Date.now() }
    
    // Update globals
    globalTrue = trueCount
    globalFalse = falseCount
    
    // Persist to DB asynchronously (fire and forget)
    persistVote(itemId, fingerprint, voteValue).catch(() => {})
    
    return NextResponse.json({ 
      userVote: { vote },
      votes: { true: trueCount, false: falseCount }
    })
  } catch (err) {
    console.error('[v0] Vote POST error:', err)
    return NextResponse.json({ error: 'Failed to submit vote' }, { status: 500 })
  }
}

async function persistVote(itemId: string, fingerprint: string, voteValue: number) {
  try {
    const sql = getSql()
    const existingVote = await sql`
      SELECT id FROM votes
      WHERE item_id = ${itemId}::uuid AND user_id = ${fingerprint}
      LIMIT 1
    `
    
    if (existingVote.length > 0) {
      await sql`
        UPDATE votes SET value = ${voteValue}
        WHERE item_id = ${itemId}::uuid AND user_id = ${fingerprint}
      `
    } else {
      await sql`
        INSERT INTO votes (id, user_id, item_id, value, created_at)
        VALUES (gen_random_uuid(), ${fingerprint}, ${itemId}::uuid, ${voteValue}, NOW())
      `
    }
  } catch (err) {
    console.error('[v0] DB persist error:', err)
  }
}
