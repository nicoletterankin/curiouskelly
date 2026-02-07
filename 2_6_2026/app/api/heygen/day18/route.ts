import { NextRequest, NextResponse } from 'next/server'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!
const HEYGEN_API = 'https://api.heygen.com'

// Kelly talking photo IDs by age and archetype
const KELLY_PHOTOS: Record<string, Record<string, string>> = {
  kid: {
    storyteller: '1024bc304a1146998bc4c360173b2c48',
    scientist: '82813816115c4fbe93b3f3f211bd9931',
    explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
  },
  adult: {
    storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
    scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
    explorer: '62516885ca4b4eae8f63b87b8c060e25',
  },
  senior: {
    storyteller: '98178c87897e4421884b535b7864ba86',
    scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
    explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
  }
}

// Day 18 Scripts - Why Cats Purr
const DAY18_SCRIPTS = {
  kid: {
    hook: "Have you ever wondered why cats make that funny rumbling sound? You know... purrrrrr. It's like they have a tiny motor inside them! But here's the mystery - scientists STILL aren't completely sure why cats purr. Want to discover what we DO know?",
    story: "A long time ago, a scientist named Champfleury studied cats and said 'The cat is a mysterious animal.' He noticed something strange - cats purr when they're happy, but ALSO when they're scared or hurt! Why would they do that? It's like if you hummed your favorite song when you were happy AND when you fell down. Weird, right?",
    wonder: "Here's something AMAZING - a cat's purr vibrates between 25 and 150 times per second. That's so fast! And scientists discovered these vibrations might actually help heal bones and muscles. Cats might be purring to make themselves feel better - like a superpower! Some people think that's why cats heal faster than dogs from injuries.",
    action: "Next time you see a cat, try this experiment! Watch when the cat purrs. Is it being petted? Is it eating? Is it alone? Write down what you notice. You'll be doing REAL science - just like the researchers who study cats!",
    wisdom: "You know what's really cool? Even the smartest scientists in the world are still asking 'why do cats purr?' Some mysteries take hundreds of years to solve. And that's okay! The best part of science isn't always finding answers - it's asking really good questions. What questions do YOU have about animals?"
  },
  adult: {
    hook: "That rhythmic rumble emanating from your cat - we've all heard it. But here's what's fascinating: after thousands of years of domestication, we still don't fully understand why cats purr. The mechanism involves rapid laryngeal muscle movement, but the PURPOSE? That remains beautifully mysterious.",
    story: "In the 1800s, naturalists began formally studying cat behavior. They documented something puzzling: cats purr not only during contentment but also during stress, illness, and even approaching death. This contradicted the simple 'happy cat' explanation. The question became: what evolutionary advantage does purring provide?",
    wonder: "The physics of purring is remarkable. Frequencies between 25-150 Hz fall within ranges that promote bone density and tissue healing. Studies suggest purring may be a self-healing mechanism. Cats confined to cages heal from bone injuries faster than dogs. Coincidence? Perhaps not. The purr might be nature's built-in physical therapy.",
    action: "Consider this observational experiment: over the next week, note the contexts in which cats purr. Time of day, social situation, health state. You're replicating the methodology scientists use. Pattern recognition in natural behavior - that's ethology in practice.",
    wisdom: "The cat's purr reminds us that familiar phenomena can harbor deep mysteries. We share our homes with these creatures, yet fundamental questions remain. As physicist Richard Feynman noted, not knowing is far more interesting than having answers that might be wrong. What everyday mysteries have you stopped questioning?"
  },
  senior: {
    hook: "After a lifetime of living with cats, you might assume we understand them completely. Yet that familiar purr - that vibration you've felt countless times - remains scientifically enigmatic. Isn't it wonderful that mystery persists in the most ordinary moments?",
    story: "Throughout history, cats have been worshipped, feared, and loved. The ancient Egyptians revered them. Medieval Europeans suspected them. Yet across all these centuries, no one definitively answered: why do cats purr? Each generation of scientists uncovers new pieces, but the complete picture eludes us still.",
    wonder: "Recent research reveals the purr's frequency range promotes healing - bone regeneration, tissue repair. Perhaps cats evolved this mechanism as a survival advantage. After all, cats spend much of their lives resting. The purr might maintain their bodies during those long, still hours. Self-care, written into their very biology.",
    action: "If you have a cat, take a moment simply to observe. Place your hand gently on their side while they purr. Feel that vibration. Consider what's happening at the cellular level - possible healing, possible communication, possible contentment. You're touching a mystery that science continues to explore.",
    wisdom: "A lifetime teaches us this: the more we learn, the more we realize remains unknown. The cat's purr is a small reminder of this truth. We need not solve every mystery to appreciate its wonder. Sometimes, sitting with a purring cat, not knowing why, is enough."
  }
}

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

async function generateVideo(talkingPhotoId: string, script: string, title: string) {
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

  const res = await fetch(`${HEYGEN_API}/v2/video/generate`, {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })

  const data = await res.json()
  return {
    success: res.ok,
    videoId: data.data?.video_id,
    error: data.error?.message,
    title
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const age = searchParams.get('age') || 'adult'
  const archetype = searchParams.get('archetype') || 'storyteller'
  const phase = searchParams.get('phase')
  const videoId = searchParams.get('video_id')

  // Check video status
  if (videoId) {
    const res = await fetch(`${HEYGEN_API}/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    })
    const data = await res.json()
    return NextResponse.json({
      videoId,
      status: data.data?.status,
      videoUrl: data.data?.video_url,
      duration: data.data?.duration
    })
  }

  const talkingPhotoId = KELLY_PHOTOS[age]?.[archetype]
  if (!talkingPhotoId) {
    return NextResponse.json({ error: `No photo ID for ${age}/${archetype}` }, { status: 400 })
  }

  const scripts = DAY18_SCRIPTS[age as keyof typeof DAY18_SCRIPTS]
  if (!scripts) {
    return NextResponse.json({ error: `No scripts for age: ${age}` }, { status: 400 })
  }

  // Generate single phase
  if (phase && PHASES.includes(phase as any)) {
    const script = scripts[phase as keyof typeof scripts]
    const result = await generateVideo(
      talkingPhotoId,
      script,
      `Day18-${age}-${archetype}-${phase}`
    )
    return NextResponse.json({
      ...result,
      checkStatus: result.videoId ? `/api/heygen/day18?video_id=${result.videoId}` : null
    })
  }

  // Generate ALL 5 phases
  const results = []
  for (const p of PHASES) {
    const script = scripts[p as keyof typeof scripts]
    const result = await generateVideo(
      talkingPhotoId,
      script,
      `Day18-${age}-${archetype}-${p}`
    )
    results.push({ phase: p, ...result })
    
    // Small delay between requests
    await new Promise(r => setTimeout(r, 500))
  }

  return NextResponse.json({
    message: `Generated all 5 phases for Day 18 - ${age} ${archetype}`,
    age,
    archetype,
    talkingPhotoId,
    results,
    checkStatus: results.map(r => r.videoId ? `/api/heygen/day18?video_id=${r.videoId}` : null)
  })
}
