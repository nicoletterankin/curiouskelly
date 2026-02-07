import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * DAY 22 HEYGEN BATCH GENERATION
 * 
 * Generates Day 22 lesson videos: "Why Ice Floats"
 * Uses database scripts and translations
 * 
 * For Day 22, we start simple:
 * - 6 languages
 * - 1 age (adult) 
 * - 1 archetype (storyteller)
 * - 5 phases
 * = 30 videos total to start
 * 
 * POST /api/heygen/day22-batch - Start batch generation
 * GET /api/heygen/day22-batch - Check progress
 */

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Kelly voice IDs by age
const VOICES: Record<string, string> = {
  kid: '3WLAUtAsXlPNi4dw1tDM',
  adult: 'BbuMXx40WT4ZuAgRXvNx',
  senior: 'RLUgsoI4JTB52m0yY4yM',
}

// Kelly talking photo IDs by age × archetype  
const LOOKS: Record<string, string> = {
  'kid-storyteller': '1024bc304a1146998bc4c360173b2c48',
  'adult-storyteller': '3d6a9d6f91b444469dae87ebb3d9eba6',
  'senior-storyteller': '98178c87897e4421884b535b7864ba86',
}

const LANGUAGES = ['en', 'es', 'fr', 'de', 'pt', 'zh']
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

// Check remaining HeyGen credits
async function checkCredits(): Promise<number> {
  const res = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY },
  })
  const data = await res.json()
  return data?.data?.remaining_quota || 0
}

// Get scripts from database
async function getScripts(dayOfYear: number, languageCode: string): Promise<Record<string, string> | null> {
  // First try translations
  if (languageCode !== 'en') {
    const trans = await sql`
      SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lesson_translations
      WHERE day_of_year = ${dayOfYear} AND language_code = ${languageCode}
      LIMIT 1
    `
    if (trans.length > 0) {
      return {
        hook: trans[0].hook_script || '',
        story: trans[0].story_script || '',
        wonder: trans[0].wonder_script || '',
        action: trans[0].action_script || '',
        wisdom: trans[0].wisdom_script || '',
      }
    }
  }
  
  // Fallback to English base lesson
  const lesson = await sql`
    SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons
    WHERE day_of_year = ${dayOfYear}
    LIMIT 1
  `
  if (lesson.length > 0) {
    return {
      hook: lesson[0].hook_script || '',
      story: lesson[0].story_script || '',
      wonder: lesson[0].wonder_script || '',
      action: lesson[0].action_script || '',
      wisdom: lesson[0].wisdom_script || '',
    }
  }
  
  return null
}

// Generate a single video
async function generateVideo(params: {
  text: string
  voiceId: string
  photoId: string
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
          character: { type: 'talking_photo', talking_photo_id: params.photoId },
          voice: { type: 'text', input_text: params.text, voice_id: params.voiceId },
        }],
        dimension: { width: 1920, height: 1080 },
        title: params.title,
      }),
    })
    
    const data = await res.json()
    if (data?.data?.video_id) {
      return { videoId: data.data.video_id, error: null }
    }
    return { videoId: null, error: data?.error?.message || 'Unknown error' }
  } catch (err) {
    return { videoId: null, error: String(err) }
  }
}

// Save to DB
async function saveToDb(params: {
  dayOfYear: number
  phase: string
  language: string
  ageCategory: string
  archetype: string
  heygenVideoId: string
  script: string
}) {
  await sql`
    INSERT INTO heygen_videos (
      id, day_of_year, phase, language, age_category, archetype,
      heygen_video_id, script, status, created_at
    ) VALUES (
      gen_random_uuid(),
      ${params.dayOfYear},
      ${params.phase},
      ${params.language},
      ${params.ageCategory},
      ${params.archetype},
      ${params.heygenVideoId},
      ${params.script},
      'processing',
      NOW()
    )
    ON CONFLICT DO NOTHING
  `
}

export async function GET() {
  const credits = await checkCredits()
  
  // Check existing Day 22 videos
  const existing = await sql`
    SELECT phase, language, status, COUNT(*) as count
    FROM heygen_videos
    WHERE day_of_year = 22
    GROUP BY phase, language, status
    ORDER BY language, phase
  `
  
  const totalNeeded = LANGUAGES.length * PHASES.length // 30 for adult-storyteller only
  const completed = existing.filter((r: Record<string, unknown>) => r.status === 'completed').reduce(
    (sum: number, r: Record<string, unknown>) => sum + Number(r.count), 0
  )
  const pending = existing.filter((r: Record<string, unknown>) => r.status === 'pending').reduce(
    (sum: number, r: Record<string, unknown>) => sum + Number(r.count), 0
  )
  
  return NextResponse.json({
    day: 22,
    title: 'Why Ice Floats',
    credits: credits,
    totalNeeded,
    completed,
    pending,
    remaining: totalNeeded - completed - pending,
    breakdown: existing,
  })
}

export async function POST(request: NextRequest) {
  const { limit = 5 } = await request.json().catch(() => ({}))
  
  const credits = await checkCredits()
  if (credits < limit) {
    return NextResponse.json({ 
      error: `Only ${credits} credits remaining, need ${limit}` 
    }, { status: 400 })
  }
  
  const results: { 
    generated: string[]
    failed: string[]
    skipped: string[]
  } = { generated: [], failed: [], skipped: [] }
  
  const age = 'adult'
  const archetype = 'storyteller'
  const photoId = LOOKS[`${age}-${archetype}`]
  const voiceId = VOICES[age]
  
  for (const lang of LANGUAGES) {
    const scripts = await getScripts(22, lang)
    if (!scripts) {
      results.failed.push(`${lang}: No scripts found`)
      continue
    }
    
    for (const phase of PHASES) {
      if (results.generated.length >= limit) {
        break
      }
      
      const key = `day22-${lang}-${age}-${archetype}-${phase}`
      const script = scripts[phase]
      
      if (!script || script.trim().length === 0) {
        results.skipped.push(`${key}: Empty script`)
        continue
      }
      
      // Check if already exists
      const existing = await sql`
        SELECT id FROM heygen_videos
        WHERE day_of_year = 22 
          AND phase = ${phase}
          AND language = ${lang}
          AND age_category = ${age}
          AND archetype = ${archetype}
        LIMIT 1
      `
      
      if (existing.length > 0) {
        results.skipped.push(`${key}: Already exists`)
        continue
      }
      
      // Generate video
      const { videoId, error } = await generateVideo({
        text: script,
        voiceId,
        photoId,
        title: key,
      })
      
      if (videoId) {
        await saveToDb({
          dayOfYear: 22,
          phase,
          language: lang,
          ageCategory: age,
          archetype,
          heygenVideoId: videoId,
          script,
        })
        results.generated.push(`${key}: ${videoId}`)
      } else {
        results.failed.push(`${key}: ${error}`)
      }
      
      // Rate limit - 2 second delay
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
    
    if (results.generated.length >= limit) {
      break
    }
  }
  
  return NextResponse.json({
    status: 'batch_started',
    results,
    creditsUsed: results.generated.length,
    message: `Generated ${results.generated.length} videos, ${results.skipped.length} skipped, ${results.failed.length} failed`,
  })
}
