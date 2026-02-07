import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
const HEYGEN_API_BASE = 'https://api.heygen.com'

// Kelly Avatar IDs - 36 combinations (3 ages × 12 archetypes)
const KELLY_AVATARS: Record<string, Record<string, string>> = {
  kid: {
    storyteller: '1024bc304a1146998bc4c360173b2c48',
    explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
    scientist: '82813816115c4fbe93b3f3f211bd9931',
    architect: 'cc1dd0e9e2fd432099985c9b036ed836',
    strategist: '6249632f58ce479891de00b4da5fb88d',
    diplomat: '48bddc41ae94473caa645ce9ab93136d',
    mystic: '5cff601bfb344015a65ff46c6b8cd70a',
    rebel: 'd4e960f7a3424d869877f3a951adfae7',
    macgyver: '7b6ab196f2c7430b945411df51a84c58',
    empath: 'deeb27f2648848b48c5c1ce59059bd54',
    provider: 'deaa213342944dc2bf671abe1442e316',
    survivor: 'bd579e4ca77444aca2bfea8ee9070830',
  },
  adult: {
    storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
    explorer: '62516885ca4b4eae8f63b87b8c060e25',
    scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
    architect: '35d0115505824e3182eb9d2ee8cfe73d',
    strategist: '08d53d1b065041bda2e5b6bc32962a8a',
    diplomat: 'c3cdbe48fe274420a7f45a4da7e366aa',
    mystic: 'dfaf9fbd644a475595b178f0be65a39a',
    rebel: '390be3fb2b064883bb2304fc3968fd87',
    macgyver: 'c5aab6ab13d940f8ae4700d546bd6b6b',
    empath: '6bb1a05678c64213a1ed3a4dc790b81e',
    provider: '9a143feeb2994989b034cebeb78753be',
    survivor: '831c8d6048104ba0b03a74a36543cfb9',
  },
  senior: {
    storyteller: '98178c87897e4421884b535b7864ba86',
    explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
    scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
    architect: '42e9197ab9d84961915b00d5cc780190',
    strategist: 'e4ab0d4d1f1b4dc9b81a1076b018557f',
    diplomat: 'a82183881e284e3782db75b755c3f080',
    mystic: 'c6d104b2ca354b0a9593cb840988bf6e',
    rebel: 'dc835263eaa247f5b0e06106b848df18',
    macgyver: 'cb5b025506284d64b696e296ca2feead',
    empath: '493dac2cf2ba4509b3cc048ff819765e',
    provider: '12582467e9ff48889d7b2435642e2d65',
    survivor: '9a143feeb2994989b034cebeb78753be',
  }
}

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
const AGES = ['kid', 'adult', 'senior'] as const
const ARCHETYPES = ['storyteller', 'explorer', 'scientist', 'architect', 'strategist', 'diplomat', 'mystic', 'rebel', 'macgyver', 'empath', 'provider', 'survivor'] as const

type Age = typeof AGES[number]
type Archetype = typeof ARCHETYPES[number]
type Phase = typeof PHASES[number]

// Get scripts from lessons table
async function getLessonScripts(dayOfYear: number): Promise<Record<string, string> | null> {
  const sql = getSql()
  const lessons = await sql`
    SELECT hook_script, story_script, wonder_script, action_script, wisdom_script, title
    FROM lessons 
    WHERE day_of_year = ${dayOfYear}
    LIMIT 1
  `
  
  if (lessons.length === 0) return null
  
  const lesson = lessons[0]
  return {
    hook: lesson.hook_script,
    story: lesson.story_script,
    wonder: lesson.wonder_script,
    action: lesson.action_script,
    wisdom: lesson.wisdom_script,
    title: lesson.title
  }
}

// Generate a single video and track in database
async function generateAndTrackVideo(
  dayOfYear: number,
  phase: Phase,
  age: Age,
  archetype: Archetype,
  script: string
): Promise<{ success: boolean; videoId?: string; dbId?: string; error?: string }> {
  
  const avatarKey = KELLY_AVATARS[age][archetype]
  const sql = getSql()
  
  // Step 1: Create database record FIRST
  const dbResult = await sql`
    INSERT INTO heygen_videos (
      day_of_year, phase, age_category, archetype,
      script, avatar_key, status, created_at, updated_at
    ) VALUES (
      ${dayOfYear}, ${phase}, ${age}, ${archetype},
      ${script}, ${avatarKey}, 'pending', NOW(), NOW()
    )
    ON CONFLICT (day_of_year, phase, age_category, archetype) 
    DO UPDATE SET script = ${script}, status = 'pending', updated_at = NOW()
    RETURNING id
  `
  
  const dbId = dbResult[0]?.id
  
  // Step 2: Call HeyGen API
  try {
    const payload = {
      video_inputs: [{
        character: {
          type: "talking_photo",
          talking_photo_id: avatarKey
        },
        voice: {
          type: "text",
          input_text: script,
          voice_id: "1bd001e7e50f421d891986aad5158bc8" // Kelly's voice
        }
      }],
      dimension: { width: 1080, height: 1920 }, // Portrait for mobile
      test: false
    }
    
    const res = await fetch(`${HEYGEN_API_BASE}/v2/video/generate`, {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY!,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })
    
    const data = await res.json()
    
    if (!res.ok || !data.data?.video_id) {
      // Update DB with error
      await sql`
        UPDATE heygen_videos 
        SET status = 'failed', error_message = ${data.error?.message || 'API error'}, updated_at = NOW()
        WHERE id = ${dbId}
      `
      return { success: false, dbId, error: data.error?.message || `API error ${res.status}` }
    }
    
    const heygenVideoId = data.data.video_id
    
    // Step 3: Update database with HeyGen video ID
    await sql`
      UPDATE heygen_videos 
      SET heygen_video_id = ${heygenVideoId}, status = 'processing', updated_at = NOW()
      WHERE id = ${dbId}
    `
    
    return { success: true, videoId: heygenVideoId, dbId }
    
  } catch (e) {
    const errorMsg = e instanceof Error ? e.message : 'Unknown error'
    await sql`
      UPDATE heygen_videos 
      SET status = 'failed', error_message = ${errorMsg}, updated_at = NOW()
      WHERE id = ${dbId}
    `
    return { success: false, dbId, error: errorMsg }
  }
}

export const maxDuration = 300

export async function POST(request: NextRequest) {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }
  
  const body = await request.json()
  const dayOfYear = body.day || 18
  const targetPhase = body.phase as Phase | undefined
  const targetAge = body.age as Age | undefined
  const targetArchetype = body.archetype as Archetype | undefined
  const batchSize = body.batch || 5
  
  // Get lesson scripts
  const scripts = await getLessonScripts(dayOfYear)
  if (!scripts) {
    return NextResponse.json({ error: `No lesson found for day ${dayOfYear}` }, { status: 404 })
  }
  
  // Build job list based on filters
  const jobs: Array<{ phase: Phase; age: Age; archetype: Archetype }> = []
  
  const phases = targetPhase ? [targetPhase] : PHASES
  const ages = targetAge ? [targetAge] : AGES
  const archetypes = targetArchetype ? [targetArchetype] : ARCHETYPES
  
  for (const phase of phases) {
    for (const age of ages) {
      for (const archetype of archetypes) {
        jobs.push({ phase, age, archetype })
      }
    }
  }
  
  const sql = getSql()
  // Check which jobs already exist and are not failed
  const existingVideos = await sql`
    SELECT phase, age_category, archetype, status
    FROM heygen_videos
    WHERE day_of_year = ${dayOfYear}
      AND status IN ('processing', 'completed')
  `
  
  const existingSet = new Set(
    existingVideos.map(v => `${v.phase}-${v.age_category}-${v.archetype}`)
  )
  
  // Filter to only jobs that need to be created
  const pendingJobs = jobs.filter(j => !existingSet.has(`${j.phase}-${j.age}-${j.archetype}`))
  
  // Take only batchSize jobs
  const batchJobs = pendingJobs.slice(0, batchSize)
  
  const results: Array<{
    phase: string
    age: string
    archetype: string
    success: boolean
    videoId?: string
    error?: string
  }> = []
  
  for (const job of batchJobs) {
    const script = scripts[job.phase]
    const result = await generateAndTrackVideo(dayOfYear, job.phase, job.age, job.archetype, script)
    
    results.push({
      phase: job.phase,
      age: job.age,
      archetype: job.archetype,
      success: result.success,
      videoId: result.videoId,
      error: result.error
    })
    
    // Rate limit: 500ms between requests
    await new Promise(resolve => setTimeout(resolve, 500))
  }
  
  const succeeded = results.filter(r => r.success).length
  const failed = results.filter(r => !r.success).length
  
  return NextResponse.json({
    success: true,
    day: dayOfYear,
    title: scripts.title,
    batch: {
      requested: batchJobs.length,
      succeeded,
      failed
    },
    progress: {
      totalJobs: jobs.length,
      alreadyExist: existingSet.size,
      remaining: pendingJobs.length - batchJobs.length
    },
    results
  })
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const dayOfYear = parseInt(searchParams.get('day') || '18')
  
  const sql = getSql()
  // Get status of all videos for this day
  const videos = await sql`
    SELECT phase, age_category, archetype, status, heygen_video_id, video_url, created_at, completed_at
    FROM heygen_videos
    WHERE day_of_year = ${dayOfYear}
    ORDER BY phase, age_category, archetype
  `
  
  const statusCounts = {
    pending: videos.filter(v => v.status === 'pending').length,
    processing: videos.filter(v => v.status === 'processing').length,
    completed: videos.filter(v => v.status === 'completed').length,
    failed: videos.filter(v => v.status === 'failed').length
  }
  
  const scripts = await getLessonScripts(dayOfYear)
  
  return NextResponse.json({
    day: dayOfYear,
    title: scripts?.title || 'Unknown',
    totalExpected: PHASES.length * AGES.length * ARCHETYPES.length, // 180
    totalInDb: videos.length,
    statusCounts,
    videos: videos.map(v => ({
      key: `${v.phase}-${v.age_category}-${v.archetype}`,
      status: v.status,
      hasUrl: !!v.video_url,
      heygenId: v.heygen_video_id
    }))
  })
}
