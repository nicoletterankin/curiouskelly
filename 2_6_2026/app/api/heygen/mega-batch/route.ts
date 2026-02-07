import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * MEGA BATCH HEYGEN GENERATOR
 * 
 * Strategy: Use all 418 credits efficiently before renewal
 * 
 * PRIORITY ORDER (highest impact first):
 * 1. Days 1-7: First week - critical for onboarding
 * 2. Days 20-28: Current week - immediate need
 * 3. Milestone days: 30, 60, 90, 100, 180, 200, 300, 365
 * 
 * VIDEO PRIORITY:
 * 1. English adult storyteller - our default experience
 * 2. Spanish adult storyteller - largest language market
 * 3. English kid/teen - age variety
 * 4. Other languages adult - expansion
 * 
 * Each video ~30 seconds = ~1 credit
 * 5 phases × 1 language × 1 age = 5 credits per complete day
 * With 418 credits: ~83 complete day×language×age combos
 */

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Kelly configurations
const VOICES: Record<string, string> = {
  kid: '3WLAUtAsXlPNi4dw1tDM',
  teen: '3WLAUtAsXlPNi4dw1tDM', // Same as kid for now
  adult: 'BbuMXx40WT4ZuAgRXvNx',
  senior: 'RLUgsoI4JTB52m0yY4yM',
}

const LOOKS: Record<string, string> = {
  'kid-storyteller': '1024bc304a1146998bc4c360173b2c48',
  'teen-storyteller': '1024bc304a1146998bc4c360173b2c48',
  'adult-storyteller': '3d6a9d6f91b444469dae87ebb3d9eba6',
  'senior-storyteller': '98178c87897e4421884b535b7864ba86',
}

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

// Priority days - most important first
const PRIORITY_DAYS = [
  // Week 1 - Critical onboarding
  1, 2, 3, 4, 5, 6, 7,
  // Current week
  20, 21, 22, 23, 24, 25, 26, 27, 28,
  // Milestone days
  30, 60, 90, 100, 180, 200, 300, 365,
  // Week 2
  8, 9, 10, 11, 12, 13, 14,
  // Rest of month 1
  15, 16, 17, 18, 19, 29, 31,
]

// Priority combos in order
const PRIORITY_COMBOS: Array<{ age: string; lang: string }> = [
  { age: 'adult', lang: 'en' },   // English adults - primary
  { age: 'adult', lang: 'es' },   // Spanish adults - huge market
  { age: 'kid', lang: 'en' },     // English kids - age variety
  { age: 'teen', lang: 'en' },    // English teens
  { age: 'adult', lang: 'fr' },   // French adults
  { age: 'adult', lang: 'de' },   // German adults
  { age: 'adult', lang: 'pt' },   // Portuguese adults
  { age: 'kid', lang: 'es' },     // Spanish kids
]

async function checkCredits(): Promise<number> {
  try {
    const res = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY },
    })
    const data = await res.json()
    return data?.data?.remaining_quota || 0
  } catch {
    return 0
  }
}

async function getScript(dayOfYear: number, phase: string, lang: string): Promise<string | null> {
  // Try translation first
  if (lang !== 'en') {
    const trans = await sql`
      SELECT ${sql(phase + '_script')} as script
      FROM lesson_translations
      WHERE day_of_year = ${dayOfYear} AND language_code = ${lang}
      LIMIT 1
    `.catch(() => [])
    if (trans.length > 0 && trans[0].script) {
      return trans[0].script
    }
  }
  
  // Fallback to base lesson (English)
  const lesson = await sql`
    SELECT ${sql(phase + '_script')} as script
    FROM lessons
    WHERE day_of_year = ${dayOfYear}
    LIMIT 1
  `.catch(() => [])
  
  return lesson[0]?.script || null
}

async function videoExists(day: number, phase: string, lang: string, age: string): Promise<boolean> {
  const existing = await sql`
    SELECT id FROM heygen_videos
    WHERE day_of_year = ${day} 
      AND phase = ${phase}
      AND language = ${lang}
      AND age_category = ${age}
    LIMIT 1
  `.catch(() => [])
  return existing.length > 0
}

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
        dimension: { width: 1080, height: 1920 }, // Portrait for mobile
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

async function saveToDb(params: {
  dayOfYear: number
  phase: string
  language: string
  ageCategory: string
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
      'storyteller',
      ${params.heygenVideoId},
      ${params.script},
      'processing',
      NOW()
    )
    ON CONFLICT DO NOTHING
  `.catch(e => console.error('DB save error:', e))
}

// GET: Show generation plan and status
export async function GET() {
  const credits = await checkCredits()
  
  // Count existing videos
  const existing = await sql`
    SELECT 
      day_of_year,
      language,
      age_category,
      COUNT(*) as phase_count,
      COUNT(*) FILTER (WHERE status = 'completed') as completed,
      COUNT(*) FILTER (WHERE status = 'processing') as processing
    FROM heygen_videos
    GROUP BY day_of_year, language, age_category
    ORDER BY day_of_year
  `
  
  // Calculate what we can generate
  const totalPossible = Math.floor(credits) // ~1 credit per video
  const existingCount = existing.reduce((sum: number, r: Record<string, unknown>) => sum + Number(r.phase_count), 0)
  
  // Priority plan
  const plan: Array<{ day: number; age: string; lang: string; phases: number }> = []
  let planned = 0
  
  for (const day of PRIORITY_DAYS) {
    for (const combo of PRIORITY_COMBOS) {
      if (planned >= totalPossible) break
      
      // Check how many phases are missing for this combo
      const existingForCombo = existing.find((e: Record<string, unknown>) => 
        e.day_of_year === day && e.language === combo.lang && e.age_category === combo.age
      )
      const phasesExisting = existingForCombo ? Number(existingForCombo.phase_count) : 0
      const phasesNeeded = 5 - phasesExisting
      
      if (phasesNeeded > 0) {
        plan.push({ day, age: combo.age, lang: combo.lang, phases: phasesNeeded })
        planned += phasesNeeded
      }
    }
    if (planned >= totalPossible) break
  }
  
  return NextResponse.json({
    credits,
    existingVideos: existingCount,
    canGenerate: totalPossible,
    plan: plan.slice(0, 50), // Show first 50 planned
    totalPlanned: planned,
    priorityDays: PRIORITY_DAYS.slice(0, 20),
    priorityCombos: PRIORITY_COMBOS,
  })
}

// POST: Execute batch generation
export async function POST(request: NextRequest) {
  const { limit = 50, dryRun = false } = await request.json().catch(() => ({}))
  
  const credits = await checkCredits()
  if (credits < 5) {
    return NextResponse.json({ error: `Only ${credits} credits remaining` }, { status: 400 })
  }
  
  const maxToGenerate = Math.min(limit, Math.floor(credits))
  
  const results: {
    generated: string[]
    skipped: string[]
    failed: string[]
    dryRun: string[]
  } = { generated: [], skipped: [], failed: [], dryRun: [] }
  
  let generated = 0
  
  // Process in priority order
  dayLoop: for (const day of PRIORITY_DAYS) {
    for (const combo of PRIORITY_COMBOS) {
      if (generated >= maxToGenerate) break dayLoop
      
      const voiceId = VOICES[combo.age]
      const photoId = LOOKS[`${combo.age}-storyteller`]
      
      if (!voiceId || !photoId) {
        results.skipped.push(`${day}-${combo.age}-${combo.lang}: Missing voice/photo config`)
        continue
      }
      
      for (const phase of PHASES) {
        if (generated >= maxToGenerate) break dayLoop
        
        const key = `day${day}-${combo.lang}-${combo.age}-${phase}`
        
        // Check if exists
        const exists = await videoExists(day, phase, combo.lang, combo.age)
        if (exists) {
          results.skipped.push(`${key}: exists`)
          continue
        }
        
        // Get script
        const script = await getScript(day, phase, combo.lang)
        if (!script || script.trim().length < 10) {
          results.skipped.push(`${key}: no script`)
          continue
        }
        
        if (dryRun) {
          results.dryRun.push(`${key}: "${script.slice(0, 50)}..."`)
          generated++
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
            dayOfYear: day,
            phase,
            language: combo.lang,
            ageCategory: combo.age,
            heygenVideoId: videoId,
            script,
          })
          results.generated.push(`${key}: ${videoId}`)
          generated++
        } else {
          results.failed.push(`${key}: ${error}`)
        }
        
        // Rate limit - wait 1.5 seconds between videos
        if (!dryRun) {
          await new Promise(resolve => setTimeout(resolve, 1500))
        }
      }
    }
  }
  
  return NextResponse.json({
    status: dryRun ? 'dry_run_complete' : 'batch_started',
    creditsUsed: generated,
    creditsRemaining: credits - generated,
    summary: {
      generated: results.generated.length,
      skipped: results.skipped.length,
      failed: results.failed.length,
    },
    results,
  })
}
