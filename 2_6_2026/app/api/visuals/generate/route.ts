import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import neon from '@/lib/neon' // Declare the neon variable here

// Visual generation API - generates educational visuals for each phase
// Uses: Fal AI for image generation, stores in Vercel Blob

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
type Phase = typeof PHASES[number]

// Phase-specific visual prompts
const PHASE_VISUAL_STYLES: Record<Phase, { style: string; mood: string }> = {
  hook: {
    style: 'Cinematic wide shot, dramatic lighting, high contrast',
    mood: 'Intriguing, mysterious, attention-grabbing'
  },
  story: {
    style: 'Educational illustration, clean lines, clear focus',
    mood: 'Narrative, engaging, exploratory'
  },
  wonder: {
    style: 'Macro photography style, detailed textures, soft bokeh',
    mood: 'Awe-inspiring, detailed, beautiful'
  },
  action: {
    style: 'Step-by-step visual, hands-on feel, bright colors',
    mood: 'Practical, achievable, motivating'
  },
  wisdom: {
    style: 'Contemplative, golden hour lighting, serene',
    mood: 'Reflective, meaningful, lasting'
  }
}

export async function POST(request: Request) {
  try {
    const { dayNumber, phase, title, topic } = await request.json()
    
    if (!dayNumber || !phase || !title) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
    }
    
    if (!PHASES.includes(phase)) {
      return NextResponse.json({ error: 'Invalid phase' }, { status: 400 })
    }
    
    const FAL_API_KEY = process.env.FAL_API_KEY
    if (!FAL_API_KEY) {
      return NextResponse.json({ error: 'FAL_API_KEY not configured' }, { status: 500 })
    }
    
    const style = PHASE_VISUAL_STYLES[phase as Phase]
    const prompt = buildVisualPrompt(title, topic || title, phase as Phase, style)
    
    // Generate image using Fal AI
    const imageResponse = await fetch('https://fal.run/fal-ai/flux/schnell', {
      method: 'POST',
      headers: {
        'Authorization': `Key ${FAL_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt,
        image_size: { width: 1280, height: 720 },
        num_images: 1,
        enable_safety_checker: true
      })
    })
    
    if (!imageResponse.ok) {
      const error = await imageResponse.text()
      console.error('[Visual Generate] Fal AI error:', error)
      return NextResponse.json({ error: 'Failed to generate image' }, { status: 500 })
    }
    
    const result = await imageResponse.json()
    const imageUrl = result.images?.[0]?.url
    
    if (!imageUrl) {
      return NextResponse.json({ error: 'No image returned' }, { status: 500 })
    }
    
    // Store in database
    await sql`
      INSERT INTO generated_assets (
        day_number, phase, asset_type, url, prompt, status, created_at
      ) VALUES (
        ${dayNumber}, ${phase}, 'visual', ${imageUrl}, ${prompt}, 'completed', NOW()
      )
      ON CONFLICT (day_number, phase, asset_type) 
      DO UPDATE SET url = ${imageUrl}, prompt = ${prompt}, status = 'completed', updated_at = NOW()
    `
    
    return NextResponse.json({
      success: true,
      dayNumber,
      phase,
      url: imageUrl,
      prompt
    })
    
  } catch (error) {
    console.error('[Visual Generate] Error:', error)
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}

function buildVisualPrompt(title: string, topic: string, phase: Phase, style: { style: string; mood: string }): string {
  const basePrompt = `Educational visual for "${title}". ${style.style}. Mood: ${style.mood}. 
Topic: ${topic}. 
Style: Professional, child-friendly, no text or words in image, photorealistic where appropriate.
Aspect ratio: 16:9 cinematic.
Quality: High resolution, sharp focus, professional lighting.`

  // Phase-specific additions
  const phaseAdditions: Record<Phase, string> = {
    hook: 'Create a visually striking opening image that sparks curiosity about the topic.',
    story: 'Illustrate the main concept being taught in a clear, educational way.',
    wonder: 'Show the beautiful details or surprising aspects of this topic.',
    action: 'Depict hands-on activity or experiment related to this lesson.',
    wisdom: 'Create a contemplative, memorable closing image that reinforces the lesson.'
  }

  return `${basePrompt}\n${phaseAdditions[phase]}`
}

// GET endpoint to check generation status
export async function GET(request: Request) {
  const url = new URL(request.url)
  const dayNumber = url.searchParams.get('day')
  const phase = url.searchParams.get('phase')
  
  if (dayNumber && phase) {
    const result = await sql`
      SELECT * FROM generated_assets 
      WHERE day_number = ${dayNumber} AND phase = ${phase} AND asset_type = 'visual'
      LIMIT 1
    `
    return NextResponse.json(result?.[0] || { status: 'not_found' })
  }
  
  // Return coverage stats
  const stats = await sql`
    SELECT phase, COUNT(*) as count 
    FROM generated_assets 
    WHERE asset_type = 'visual' AND status = 'completed'
    GROUP BY phase
  `
  
  return NextResponse.json({
    coverage: Object.fromEntries(stats.map(s => [s.phase, parseInt(s.count)])),
    target: {
      hook: 365,
      story: 365,
      wonder: 365,
      action: 365,
      wisdom: 365
    }
  })
}
