import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY
const HEYGEN_API_BASE = 'https://api.heygen.com'

// All 36 Kelly avatar configurations (3 ages × 12 archetypes)
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

// Day 18 Scripts - Why Cats Purr
const DAY18_SCRIPTS: Record<string, Record<string, string>> = {
  kid: {
    hook: "Have you ever wondered why cats purr? That rumbling sound isn't just because they're happy - it's one of nature's coolest mysteries!",
    story: "Scientists discovered that cats purr at a special frequency - between 25 and 150 vibrations per second. That's the same frequency that helps bones heal faster! Cats might actually be giving themselves a healing massage every time they purr.",
    wonder: "Here's the amazing part - cats purr when they're happy, but also when they're scared or hurt. It's like they have a built-in healing superpower that works no matter how they feel!",
    action: "Try this: Next time you're near a cat, gently place your hand on their chest while they purr. Can you feel those vibrations? You're feeling actual healing frequencies!",
    wisdom: "Even the famous scientist Sigmund Freud loved cats. He said time spent with cats is never wasted. Maybe cats teach us that taking time to rest and heal ourselves isn't lazy - it's smart!"
  },
  adult: {
    hook: "You've heard cats purr a thousand times, but do you know why? The answer involves healing frequencies, self-medication, and one of biology's most elegant mysteries.",
    story: "Research shows cats purr at frequencies between 25 and 150 Hertz - precisely the range that promotes bone density and tissue healing. Unlike other vocalizations, purring happens on both inhale and exhale, creating a continuous vibration therapy.",
    wonder: "What's remarkable is that cats purr in all emotional states - contentment, stress, even approaching death. This suggests purring isn't just communication; it's a biological self-repair mechanism that evolved independently of emotional expression.",
    action: "Here's something to try: When you're feeling stressed, try humming at a low frequency, similar to a cat's purr. Studies suggest these vibrations can lower blood pressure and reduce anxiety in humans too.",
    wisdom: "Freud kept a cat named Jofi in his therapy sessions, believing the cat could sense his patients' emotional states. Perhaps cats understand something we're still learning - that sometimes the most healing thing we can do is simply be present and vibrate at the right frequency."
  },
  senior: {
    hook: "After all these years of listening to cats purr, science has finally revealed why - and it turns out these ancient companions have been practicing medicine longer than humans have.",
    story: "The purring frequency of 25 to 150 Hertz corresponds exactly to frequencies used in modern medicine to promote bone healing and reduce inflammation. Cats essentially carry a portable healing device in their throats.",
    wonder: "Most fascinating is that cats purr during birth, death, and everything between. This isn't emotional expression - it's survival. A cat lying still while healing can't exercise, so purring provides the vibrational stimulation their bodies need.",
    action: "If you have a cat, spend a few quiet minutes with them today. Place your hand gently on their body while they purr. You're not just bonding - you may be receiving some of those healing vibrations yourself.",
    wisdom: "Freud observed that cats live entirely in the present moment, neither regretting the past nor anxious about the future. In our later years, perhaps that's the deepest lesson cats offer - the wisdom of presence, wrapped in a healing purr."
  }
}

type Age = typeof AGES[number]
type Archetype = typeof ARCHETYPES[number]
type Phase = typeof PHASES[number]

// Build flat list of all 180 combinations
function buildJobList(): Array<{ index: number; age: Age; archetype: Archetype; phase: Phase }> {
  const jobs: Array<{ index: number; age: Age; archetype: Archetype; phase: Phase }> = []
  let index = 0
  for (const age of AGES) {
    for (const archetype of ARCHETYPES) {
      for (const phase of PHASES) {
        jobs.push({ index, age, archetype, phase })
        index++
      }
    }
  }
  return jobs
}

async function generateVideo(age: Age, archetype: Archetype, phase: Phase): Promise<{ videoId?: string; error?: string }> {
  const talkingPhotoId = KELLY_AVATARS[age][archetype]
  const script = DAY18_SCRIPTS[age][phase]
  
  const payload = {
    video_inputs: [{
      character: {
        type: "talking_photo",
        talking_photo_id: talkingPhotoId
      },
      voice: {
        type: "text",
        input_text: script,
        voice_id: "1bd001e7e50f421d891986aad5158bc8"
      }
    }],
    dimension: { width: 1280, height: 720 },
    test: false
  }
  
  try {
    const res = await fetch(`${HEYGEN_API_BASE}/v2/video/generate`, {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY!,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })
    
    const data = await res.json()
    
    if (!res.ok) {
      return { error: data.error?.message || `API error ${res.status}` }
    }
    
    return { videoId: data.data?.video_id }
  } catch (e) {
    return { error: e instanceof Error ? e.message : 'Unknown error' }
  }
}

export const maxDuration = 300 // 5 minutes max for Vercel

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  const startIndex = parseInt(searchParams.get('start') || '0')
  const batchSize = parseInt(searchParams.get('batch') || '10') // Generate 10 at a time
  
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }
  
  const allJobs = buildJobList()
  
  // Generate a batch starting from startIndex
  if (action === 'generate') {
    const endIndex = Math.min(startIndex + batchSize, allJobs.length)
    const batch = allJobs.slice(startIndex, endIndex)
    
    const results: Array<{
      index: number
      age: string
      archetype: string
      phase: string
      videoId?: string
      error?: string
    }> = []
    
    for (const job of batch) {
      const result = await generateVideo(job.age, job.archetype, job.phase)
      results.push({
        index: job.index,
        age: job.age,
        archetype: job.archetype,
        phase: job.phase,
        videoId: result.videoId,
        error: result.error
      })
      
      // 300ms delay between requests
      await new Promise(resolve => setTimeout(resolve, 300))
    }
    
    const completed = results.filter(r => r.videoId).length
    const failed = results.filter(r => r.error).length
    const nextStart = endIndex < allJobs.length ? endIndex : null
    
    return NextResponse.json({
      success: true,
      batch: {
        start: startIndex,
        end: endIndex,
        completed,
        failed
      },
      progress: {
        current: endIndex,
        total: allJobs.length,
        percentComplete: Math.round((endIndex / allJobs.length) * 100)
      },
      nextBatch: nextStart !== null ? `/api/heygen/batch-day18?action=generate&start=${nextStart}&batch=${batchSize}` : null,
      results
    })
  }
  
  // Show status/info
  return NextResponse.json({
    message: "Day 18 Batch Video Generator - 180 videos (Why Cats Purr)",
    totalVideos: allJobs.length,
    ages: AGES,
    archetypes: ARCHETYPES,
    phases: PHASES,
    usage: {
      generateBatch: "/api/heygen/batch-day18?action=generate&start=0&batch=10",
      customStart: "/api/heygen/batch-day18?action=generate&start=50&batch=20",
      description: "Call repeatedly with incrementing 'start' parameter to generate all 180 videos"
    },
    jobs: allJobs.slice(0, 5).map(j => `${j.index}: ${j.age}/${j.archetype}/${j.phase}`)
  })
}

export async function POST(request: NextRequest) {
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({ error: 'HEYGEN_API_KEY not configured' }, { status: 500 })
  }
  
  // Parse request body for start/batch params
  let startIndex = 0
  let batchSize = 15
  
  try {
    const body = await request.json()
    startIndex = body.start || 0
    batchSize = body.batch || 15
  } catch {
    // Use defaults
  }
  
  const allJobs = buildJobList()
  const endIndex = Math.min(startIndex + batchSize, allJobs.length)
  const batch = allJobs.slice(startIndex, endIndex)
  
  const results: Array<{
    index: number
    age: string
    archetype: string
    phase: string
    videoId?: string
    error?: string
  }> = []
  
  for (const job of batch) {
    const result = await generateVideo(job.age, job.archetype, job.phase)
    results.push({
      index: job.index,
      age: job.age,
      archetype: job.archetype,
      phase: job.phase,
      videoId: result.videoId,
      error: result.error
    })
    
    // 300ms delay
    await new Promise(resolve => setTimeout(resolve, 300))
  }
  
  const completed = results.filter(r => r.videoId).length
  const failed = results.filter(r => r.error).length
  
  return NextResponse.json({
    success: true,
    batch: { start: startIndex, end: endIndex, completed, failed },
    progress: {
      current: endIndex,
      total: allJobs.length,
      percentComplete: Math.round((endIndex / allJobs.length) * 100),
      remaining: allJobs.length - endIndex
    },
    nextBatch: endIndex < allJobs.length ? { start: endIndex, batch: batchSize } : null,
    results
  })
}
