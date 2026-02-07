import { NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { 
  HEYGEN_ADULT_KELLY_LOOKS, 
  HEYGEN_KID_KELLY_LOOKS, 
  HEYGEN_SENIOR_KELLY_LOOKS,
  HEYGEN_VOICES,
  SUPPORTED_LANGUAGES,
  LESSON_SEGMENTS,
} from '@/lib/kelly-assets'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Priority order: Today/Tomorrow first, then sequential through entire year
const PRIORITY_DAYS = [
  23, 24, 21, 22, // Current broadcast window
  1, 2, 3, 4, 5, 6, 7, // Week 1
  ...Array.from({length: 358}, (_, i) => i + 8).filter(d => ![21,22,23,24].includes(d)) // Rest of year
]

// Priority combos - storyteller first (primary experience), then other archetypes
const PRIORITY_ARCHETYPES = ['storyteller', 'scientist', 'explorer', 'rebel', 'mystic', 'macgyver']
const PRIORITY_AGES: Array<'adult' | 'kid' | 'senior'> = ['adult', 'kid', 'senior']
const PRIORITY_LANGS = ['en', 'es', 'fr', 'de', 'pt', 'zh']

const LOOK_MAPS = {
  kid: HEYGEN_KID_KELLY_LOOKS,
  adult: HEYGEN_ADULT_KELLY_LOOKS,
  senior: HEYGEN_SENIOR_KELLY_LOOKS,
}

async function checkCredits(): Promise<number> {
  try {
    const res = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY },
    })
    const data = await res.json()
    const quota = data?.data?.remaining_quota
    // If quota is very high (10K+), treat as unlimited
    if (quota > 10000) return 999999
    return quota || 999999 // Default to unlimited if API fails - HeyGen may have granted unlimited
  } catch {
    return 999999 // Assume unlimited - fire everything
  }
}

async function getScript(day: number, phase: string, lang: string): Promise<string | null> {
  const phaseCol = `${phase}_script`
  const sql = getSql()
  
  // Try translation first
  if (lang !== 'en') {
    const trans = await sql`
      SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lesson_translations
      WHERE day_of_year = ${day} AND language_code = ${lang}
      LIMIT 1
    `.catch(() => [])
    if (trans.length > 0 && trans[0][phaseCol]) {
      return trans[0][phaseCol] as string
    }
  }
  
  // Base English
  const lesson = await sql`
    SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons WHERE day_of_year = ${day} LIMIT 1
  `.catch(() => [])
  
  return (lesson[0]?.[phaseCol] as string) || null
}

async function videoExists(day: number, phase: string, lang: string, age: string, archetype: string): Promise<boolean> {
  try {
    const sql = getSql()
    const existing = await sql`
      SELECT id FROM heygen_videos
      WHERE day_of_year = ${day} 
        AND phase = ${phase}
        AND age_category = ${age}
        AND archetype = ${archetype}
        AND status IN ('queued', 'processing', 'completed')
      LIMIT 1
    `
    // Also check language match if column exists
    return existing.length > 0
  } catch (e) {
    console.error('videoExists error:', e)
    return false // Assume doesn't exist, will be caught by ON CONFLICT
  }
}

async function submitToHeyGen(params: {
  text: string
  voiceId: string
  lookId: string
  title: string
}): Promise<{ videoId: string | null; error: string | null }> {
  try {
    const res = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_inputs: [{
          character: { type: 'talking_photo', talking_photo_id: params.lookId },
          voice: { type: 'text', input_text: params.text, voice_id: params.voiceId },
        }],
        dimension: { width: 1080, height: 1920 },
        title: params.title,
      }),
    })
    
    const data = await res.json()
    if (data?.data?.video_id) {
      return { videoId: data.data.video_id, error: null }
    }
    return { videoId: null, error: data?.error?.message || JSON.stringify(data) }
  } catch (err) {
    return { videoId: null, error: String(err) }
  }
}

// Note: We no longer need saveToDb - we just UPDATE existing queued records
// The queue was pre-populated with all videos via neon_run_sql

// GET: Show status and plan
export async function GET() {
  const credits = await checkCredits()
  const sql = getSql()
  
  const stats = await sql`
    SELECT status, COUNT(*) as count FROM heygen_videos GROUP BY status
  `
  
  const byDay = await sql`
    SELECT day_of_year, COUNT(*) as count, 
           COUNT(*) FILTER (WHERE status = 'completed') as completed
    FROM heygen_videos 
    WHERE day_of_year IN (21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7)
    GROUP BY day_of_year ORDER BY day_of_year
  `
  
  return NextResponse.json({
    credits,
    stats,
    byDay,
    plan: {
      priorityDays: PRIORITY_DAYS,
      priorityArchetypes: PRIORITY_ARCHETYPES,
      priorityAges: PRIORITY_AGES,
      priorityLangs: PRIORITY_LANGS,
      videosPerFullDay: PRIORITY_ARCHETYPES.length * PRIORITY_AGES.length * PRIORITY_LANGS.length * LESSON_SEGMENTS.length,
    }
  })
}

// POST: FIRE ALL QUEUED VIDEOS
export async function POST(req: Request) {
  const { batchSize = 100 } = await req.json().catch(() => ({}))
  
  const startCredits = await checkCredits()
  console.log(`[FIRE] Starting with ${startCredits} credits, batch size ${batchSize}`)

  const results = {
    submitted: [] as string[],
    failed: [] as string[],
    creditsUsed: 0,
  }

  const sql = getSql()
  // Get queued videos in priority order (by day_of_year, prioritize today/tomorrow)
  // Note: heygen_videos table does NOT have 'language' column - use 'en' as default
  const queued = await sql`
    SELECT id, day_of_year, phase, age_category, archetype, script
    FROM heygen_videos
    WHERE status = 'queued'
    ORDER BY 
      CASE 
        WHEN day_of_year = 23 THEN 0
        WHEN day_of_year = 24 THEN 1
        WHEN day_of_year = 21 THEN 2
        WHEN day_of_year = 22 THEN 3
        WHEN day_of_year <= 7 THEN 4
        ELSE day_of_year + 100
      END,
      phase
    LIMIT ${batchSize}
  `

  console.log(`[FIRE] Found ${queued.length} queued videos to process`)

  for (const video of queued) {
    const age = video.age_category as 'adult' | 'kid' | 'senior'
    const archetype = video.archetype as string
    const lang = 'en' // Default to English - table doesn't have language column
    const key = `day${video.day_of_year}_${lang}_${age}_${archetype}_${video.phase}`
    
    // Get lookId
    const lookId = LOOK_MAPS[age]?.[archetype]
    if (!lookId) {
      results.failed.push(`${key}:no_look`)
      continue
    }
    
    // Get voiceId
    const voiceId = HEYGEN_VOICES[age]
    const script = video.script as string
    
    if (!script || script.length < 10) {
      results.failed.push(`${key}:no_script`)
      continue
    }
    
    // FIRE!
    const { videoId, error } = await submitToHeyGen({
      text: script,
      voiceId,
      lookId,
      title: key,
    })
    
    if (videoId) {
      // Update status to processing
      await sql`
        UPDATE heygen_videos 
        SET status = 'processing', heygen_video_id = ${videoId}, updated_at = NOW()
        WHERE id = ${video.id}
      `
      results.submitted.push(`${key}:${videoId}`)
      results.creditsUsed++
    } else {
      // Mark as failed
      await sql`
        UPDATE heygen_videos 
        SET status = 'failed', error_message = ${error}, updated_at = NOW()
        WHERE id = ${video.id}
      `
      results.failed.push(`${key}:${error}`)
    }
    
    // Rate limit - 300ms between requests (faster for batch)
    await new Promise(r => setTimeout(r, 300))
  }

  // Get remaining queue count
  const remaining = await sql`SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'queued'`

  return NextResponse.json({
    status: 'FIRED',
    batchSize,
    creditsUsed: results.creditsUsed,
    summary: {
      submitted: results.submitted.length,
      failed: results.failed.length,
      remainingInQueue: Number(remaining[0]?.count || 0),
    },
    submitted: results.submitted.slice(0, 20),
    failed: results.failed.slice(0, 20),
  })
}
