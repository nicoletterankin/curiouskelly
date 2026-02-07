import { NextRequest, NextResponse } from 'next/server'

const FAL_KEY = process.env.FAL_KEY!

// Day 20: January 20, 2026 - "Why We Yawn"
const DAY = 20
const TITLE = 'Why We Yawn'

// All 5 phases with their scripts
const PHASES = {
  hook: `Have you noticed that yawns are contagious? When someone yawns, you might suddenly feel the urge to yawn too! Scientists think this happens because of mirror neurons in our brain that help us connect with others. But here's the really curious part - yawning isn't just about being tired. Your brain actually uses yawning to cool itself down, like a built-in air conditioner! Today we're going to explore why we yawn and what it tells us about how our amazing brains work.`,
  story: `Meet Maya, a curious 10-year-old who noticed something strange during her morning class. Her teacher yawned, and suddenly half the class started yawning too, including Maya! She wondered if yawning was like a secret signal spreading through the room. That night, Maya did an experiment - she fake-yawned in front of her dog, and guess what? Her dog yawned back! Maya discovered that yawning connects us to other beings, showing empathy and understanding without saying a word.`,
  wonder: `Here's what makes yawning truly fascinating: your brain heats up when you're thinking hard or feeling tired, and a big yawn brings cool air into your mouth and nose, which cools the blood going to your brain! It's like your body's natural reset button. Scientists have found that people yawn more in rooms that are warm and less in cool rooms. Even babies yawn in the womb before they're born! What do you think your yawns are telling you about how your brain is feeling?`,
  action: `Try this yawn experiment: Tomorrow, count how many times you yawn and write down what you were doing each time. Were you tired? Bored? Did someone else just yawn? See if you can spot patterns. Then try the "contagious yawn test" - yawn near friends and family and see how many catch it. People who catch yawns easily often score higher on empathy tests! Notice if reading this made you yawn - that counts too!`,
  wisdom: `Yawning teaches us that our bodies are constantly communicating, even without words. When you catch a yawn from someone, your brain is saying "I understand you, I connect with you." This simple act shows how deeply we're wired to bond with others. Your body has its own wisdom - it knows when to cool your brain, when to take a deep breath, and when to show others that you're paying attention. Trust your body's signals, they're smarter than you might think!`
}

// Kelly Adult Storyteller image URL for lip-sync
const KELLY_IMAGE_URL = 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly/archetypes/storyteller.png'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  const phase = (searchParams.get('phase') || 'hook') as keyof typeof PHASES
  const requestId = searchParams.get('request_id')

  if (!FAL_KEY) {
    return NextResponse.json({
      error: 'FAL_KEY not configured',
      fix: 'Add FAL_KEY to environment variables'
    }, { status: 500 })
  }

  // Check status of existing generation
  if (action === 'status' && requestId) {
    const res = await fetch(`https://queue.fal.run/fal-ai/sadtalker/requests/${requestId}/status`, {
      headers: { 'Authorization': `Key ${FAL_KEY}` }
    })
    const data = await res.json()
    
    // If completed, get the result
    if (data.status === 'COMPLETED') {
      const resultRes = await fetch(`https://queue.fal.run/fal-ai/sadtalker/requests/${requestId}`, {
        headers: { 'Authorization': `Key ${FAL_KEY}` }
      })
      const result = await resultRes.json()
      
      return NextResponse.json({
        requestId,
        status: 'completed',
        videoUrl: result.video?.url,
        raw: result
      })
    }
    
    return NextResponse.json({
      requestId,
      status: data.status,
      raw: data
    })
  }

  // Queue all phases
  if (action === 'queue-all') {
    const results = []
    
    for (const [phaseName, script] of Object.entries(PHASES)) {
      try {
        // First generate audio with ElevenLabs or use existing
        // For now, we'll submit to fal with text-to-speech
        const payload = {
          source_image_url: KELLY_IMAGE_URL,
          driven_audio_url: null, // Will need audio file
          input_text: script,
          preprocess: 'crop',
          still_mode: false,
          expression_scale: 1.0
        }
        
        const res = await fetch('https://queue.fal.run/fal-ai/sadtalker', {
          method: 'POST',
          headers: {
            'Authorization': `Key ${FAL_KEY}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        })
        
        const data = await res.json()
        results.push({
          phase: phaseName,
          requestId: data.request_id,
          status: res.ok ? 'queued' : 'failed',
          error: res.ok ? null : data
        })
      } catch (err) {
        results.push({
          phase: phaseName,
          status: 'error',
          error: String(err)
        })
      }
    }
    
    return NextResponse.json({
      day: DAY,
      title: TITLE,
      queued: results.filter(r => r.status === 'queued').length,
      failed: results.filter(r => r.status !== 'queued').length,
      results
    })
  }

  // Generate single phase
  if (action === 'generate') {
    const script = PHASES[phase]
    if (!script) {
      return NextResponse.json({ error: `Invalid phase: ${phase}` }, { status: 400 })
    }

    // For lip-sync, we need an audio file. Let's use fal's wav2lip model instead
    // which can take text input directly
    const payload = {
      source_image_url: KELLY_IMAGE_URL,
      input_text: script,
      preprocess: 'crop',
      still_mode: false,
      expression_scale: 1.2,
      pose_style: 0
    }

    console.log('[fal Day20] Generating video for phase:', phase)
    console.log('[fal Day20] Payload:', JSON.stringify(payload, null, 2))

    const res = await fetch('https://queue.fal.run/fal-ai/sadtalker', {
      method: 'POST',
      headers: {
        'Authorization': `Key ${FAL_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })

    const data = await res.json()

    if (!res.ok) {
      return NextResponse.json({
        success: false,
        error: 'fal.ai API error',
        status: res.status,
        response: data,
        payload_sent: payload
      }, { status: res.status })
    }

    return NextResponse.json({
      success: true,
      requestId: data.request_id,
      status: 'queued',
      check_status: `/api/fal/day20?action=status&request_id=${data.request_id}`,
      message: 'Video queued! Check status in 30-60 seconds.',
      config: {
        day: DAY,
        phase,
        title: TITLE,
        script_length: script.length
      }
    })
  }

  // Default: show instructions
  return NextResponse.json({
    endpoint: '/api/fal/day20',
    day: DAY,
    title: TITLE,
    date: 'January 20, 2026',
    phases: Object.keys(PHASES),
    actions: {
      generate_single: '/api/fal/day20?action=generate&phase=hook',
      queue_all: '/api/fal/day20?action=queue-all',
      status: '/api/fal/day20?action=status&request_id=REQUEST_ID'
    },
    scripts: Object.fromEntries(
      Object.entries(PHASES).map(([k, v]) => [k, v.substring(0, 80) + '...'])
    )
  })
}
