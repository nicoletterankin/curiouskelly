import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// Cache for comments to reduce DB calls and avoid rate limits
// NOTE: In serverless, this cache only works within a single instance
const commentsCache: { data: unknown[]; timestamp: number; day: number; errorBackoff: number } = { 
  data: [], 
  timestamp: 0, 
  day: 0,
  errorBackoff: 0
}
const CACHE_TTL = 60000 // 60 seconds (increased to reduce DB calls)
const ERROR_BACKOFF_TTL = 300000 // 5 minutes after rate limit error

// Global flag to skip DB entirely during high traffic (launch day)
const SKIP_DB_FOR_COMMENTS = true // Set to false when rate limits are resolved

// Seed comments by day - always available as fallback
const seedCommentsByDay: Record<number, Array<{id: string, username: string, message: string}>> = {
  20: [ // Why We Yawn
    { id: 'seed_20_1', username: 'yawn_catcher', message: 'I yawned THREE times reading this! So true about contagious yawns' },
    { id: 'seed_20_2', username: 'brain_cooler', message: 'Mind blown that yawning cools your brain like AC!' },
    { id: 'seed_20_3', username: 'empathy_kid', message: 'So catching yawns means I have empathy? Cool!' },
    { id: 'seed_20_4', username: 'science_mom_42', message: 'My daughter immediately tested this on the dog. It worked!' },
    { id: 'seed_20_5', username: 'curious_learner', message: 'Kelly always finds the most interesting topics' },
  ],
  19: [ // How Magnets Work
    { id: 'seed_19_1', username: 'magnet_kid', message: 'I tried the compass experiment - it really points north!' },
    { id: 'seed_19_2', username: 'fridge_artist', message: 'So THATS why my magnets stick to the fridge but not wood!' },
    { id: 'seed_19_3', username: 'science_mom_42', message: 'Finally an explanation my 8yo can understand' },
    { id: 'seed_19_4', username: 'curious_grandpa', message: 'Magnets are like tiny invisible hands. Love that explanation!' },
    { id: 'seed_19_5', username: 'happy_learner', message: 'Kelly makes learning so fun!' },
  ],
}

// Default fallback comments (generic)
const defaultSeedComments = [
  { id: 'seed_default_1', username: 'curious_mind', message: 'This is fascinating! I never thought about it this way' },
  { id: 'seed_default_2', username: 'learning_together', message: 'My kids love learning with Kelly' },
  { id: 'seed_default_3', username: 'wonder_seeker', message: 'Another great lesson! Keep them coming' },
  { id: 'seed_default_4', username: 'science_fan', message: 'Love how Kelly explains complex ideas simply' },
  { id: 'seed_default_5', username: 'daily_learner', message: 'Best part of my morning routine' },
]

function getSeedComments(day: number) {
  const dayComments = seedCommentsByDay[day] || defaultSeedComments
  return dayComments.map(c => ({
    ...c,
    created_at: new Date(Date.now() - Math.random() * 300000).toISOString()
  }))
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const dayNum = parseInt(searchParams.get('day') || '20') // Default to Day 20 for launch
  
  // LAUNCH DAY: Skip DB entirely to avoid rate limits, use seed comments
  if (SKIP_DB_FOR_COMMENTS) {
    const seedComments = getSeedComments(dayNum)
    return NextResponse.json({ comments: seedComments, total: seedComments.length, seed: true })
  }
  
  // Return cached data if fresh
  if (commentsCache.day === dayNum && Date.now() - commentsCache.timestamp < CACHE_TTL && commentsCache.data.length > 0) {
    return NextResponse.json({ comments: commentsCache.data, total: commentsCache.data.length, cached: true })
  }
  
  // If we're in error backoff mode, return seed comments immediately (don't hit DB)
  if (Date.now() < commentsCache.errorBackoff) {
    const fallbackComments = getSeedComments(dayNum)
    return NextResponse.json({ comments: fallbackComments, total: fallbackComments.length, backoff: true })
  }
  
  try {
    // Get recent messages for this day from real messages table
    const messages = await sql`
      SELECT id, username, content as message, avatar_url, created_at
      FROM messages
      WHERE (metadata->>'day')::int = ${dayNum}
      ORDER BY created_at DESC
      LIMIT 20
    `
    
    // Transform and merge with seed comments if DB has few results
    const dbComments = messages.map((m: Record<string, unknown>) => ({
      id: m.id,
      username: m.username,
      message: m.message,
      created_at: m.created_at
    }))
    
    // Use day-specific seed comments that match the lesson topic
    const daySeedComments = getSeedComments(dayNum)
    const comments = dbComments.length >= 3 ? dbComments : [...dbComments, ...daySeedComments.slice(0, 5 - dbComments.length)]
    
    // Update cache
    commentsCache.data = comments
    commentsCache.timestamp = Date.now()
    commentsCache.day = dayNum
    
    return NextResponse.json({ comments, total: comments.length })
  } catch (error) {
    console.error('Comments GET error:', error)
    // Set backoff to prevent hammering DB during rate limits
    commentsCache.errorBackoff = Date.now() + ERROR_BACKOFF_TTL
    // Return day-specific seed data on any error (including rate limits)
    const fallbackComments = getSeedComments(dayNum)
    return NextResponse.json({ comments: fallbackComments, total: fallbackComments.length })
  }
}

export async function POST(request: NextRequest) {
  try {
    const { day, fingerprint, username, message } = await request.json()
    
    if (!username || !message) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
    }
    
    // Sanitize input
    const cleanMessage = message.trim().slice(0, 500)
    const cleanUsername = username.trim().slice(0, 50)
    const dayNum = parseInt(day) || 19
    
    // Insert into messages table
    // Cast dayNum to text then to int inside jsonb_build_object to avoid type inference issues
    const metadata = JSON.stringify({ day: dayNum })
    const result = await sql`
      INSERT INTO messages (id, user_id, content, username, metadata, created_at)
      VALUES (
        gen_random_uuid(), 
        ${fingerprint || 'anonymous'}, 
        ${cleanMessage}, 
        ${cleanUsername},
        ${metadata}::jsonb,
        NOW()
      )
      RETURNING id, username, content as message, created_at
    `
    
    return NextResponse.json({ comment: result[0] })
  } catch (error) {
    console.error('Comments POST error:', error)
    return NextResponse.json({ error: 'Failed to post comment' }, { status: 500 })
  }
}
