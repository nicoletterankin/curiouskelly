/**
 * ZIGGURAT/LAGUNA RIDGE VIDEO AUDIO GENERATION
 * 
 * Generates all 16 audio segments for the Ziggurat pitch video using Kelly's voice.
 * Uses ElevenLabs with the same voice ID as the Daily Lesson.
 * 
 * POST /api/video/ziggurat-audio
 * - Generates all 16 segments
 * - Uploads each to Vercel Blob
 * - Returns URLs for all segments
 * 
 * GET /api/video/ziggurat-audio
 * - Returns status of existing segments
 */

import { NextRequest, NextResponse } from 'next/server'
import { put, list } from '@vercel/blob'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!
// Kelly's voice ID - wAdymQH5YucAkXwmrdL0 (kelly2)
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0'

// The approved 16-segment script for Ziggurat pitch
const ZIGGURAT_SCRIPT = [
  { id: '01', text: "Let me show you something.", pause: 1.0 },
  { id: '02', text: "In nineteen sixty-eight, architect William Pereira—the man who designed the Transamerica Pyramid—was commissioned to build something unprecedented.", pause: 0.5 },
  { id: '03', text: "A facility for seven thousand engineers to manufacture the future.", pause: 0.5 },
  { id: '04', text: "Ninety-two acres. One million square feet. Four miles from the Pacific.", pause: 0.5 },
  { id: '05', text: "The company that commissioned it never moved in. The contract fell through before the doors ever opened.", pause: 0.5 },
  { id: '06', text: "So the federal government used it. For paperwork. Fifty years of paperwork.", pause: 0.8 },
  { id: '07', text: "That's not what this building was made for.", pause: 1.0 },
  { id: '08', text: "Now they're selling it. Three auctions. No deal has closed. The last bidder planned to tear it down.", pause: 0.5 },
  { id: '09', text: "Our founder sees something different.", pause: 0.8 },
  { id: '10', text: "Eighteen-inch raised floors—perfect for data centers and manufacturing. Redundant power systems. Loading docks built for high-volume logistics.", pause: 0.5 },
  { id: '11', text: "Every problem the government lists is exactly the infrastructure Lesson of the Day needs to deliver on SDG 4.", pause: 0.5 },
  { id: '12', text: "This is where iLearn computers should be manufactured. This is where global curriculum should be printed. This is where The Daily Lesson broadcasts to the world.", pause: 0.5 },
  { id: '13', text: "William Pereira designed this building to manufacture the future.", pause: 0.5 },
  { id: '14', text: "We're going to finish what he started.", pause: 1.0 },
  { id: '15', text: "Will you join us?", pause: 1.5 },
  { id: '16', text: "Lesson of the Day.", pause: 0.0 },
]

// Generate a single audio segment
async function generateSegment(segment: typeof ZIGGURAT_SCRIPT[0]): Promise<{
  id: string
  url: string
  duration: number
  characters: number
}> {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: segment.text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.3, // Slight style for documentary feel
          use_speaker_boost: true,
        },
      }),
    }
  )

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`ElevenLabs error for segment ${segment.id}: ${response.status} - ${error}`)
  }

  const audioBuffer = await response.arrayBuffer()
  
  // Upload to Vercel Blob
  const filename = `video/ziggurat/segment_${segment.id}.mp3`
  const blob = await put(filename, Buffer.from(audioBuffer), {
    access: 'public',
    contentType: 'audio/mpeg',
  })

  // Estimate duration: ~150 words per minute, average word length 5 chars
  const words = segment.text.split(' ').length
  const estimatedDuration = (words / 150) * 60

  return {
    id: segment.id,
    url: blob.url,
    duration: estimatedDuration,
    characters: segment.text.length,
  }
}

// GET - Check status of existing segments
export async function GET() {
  try {
    const blobs = await list({ prefix: 'video/ziggurat/' })
    
    const segments = ZIGGURAT_SCRIPT.map(seg => {
      const existing = blobs.blobs.find(b => b.pathname.includes(`segment_${seg.id}`))
      return {
        id: seg.id,
        text: seg.text.substring(0, 50) + (seg.text.length > 50 ? '...' : ''),
        generated: !!existing,
        url: existing?.url || null,
      }
    })

    const generatedCount = segments.filter(s => s.generated).length
    
    return NextResponse.json({
      status: generatedCount === 16 ? 'complete' : generatedCount > 0 ? 'partial' : 'pending',
      generated: generatedCount,
      total: 16,
      segments,
      script: ZIGGURAT_SCRIPT,
    })
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}

// POST - Generate all segments
export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const forceRegenerate = searchParams.get('force') === 'true'
  const singleSegment = searchParams.get('segment') // Generate only one segment
  
  try {
    const results: Array<{
      id: string
      url: string
      duration: number
      characters: number
      status: 'generated' | 'skipped' | 'error'
      error?: string
    }> = []

    // Check existing if not forcing regeneration
    let existingBlobs: string[] = []
    if (!forceRegenerate) {
      const blobs = await list({ prefix: 'video/ziggurat/' })
      existingBlobs = blobs.blobs.map(b => b.pathname)
    }

    const segmentsToGenerate = singleSegment 
      ? ZIGGURAT_SCRIPT.filter(s => s.id === singleSegment)
      : ZIGGURAT_SCRIPT

    for (const segment of segmentsToGenerate) {
      const expectedPath = `video/ziggurat/segment_${segment.id}.mp3`
      
      // Skip if already exists (unless forcing)
      if (!forceRegenerate && existingBlobs.some(p => p.includes(`segment_${segment.id}`))) {
        const existingBlob = (await list({ prefix: 'video/ziggurat/' })).blobs.find(
          b => b.pathname.includes(`segment_${segment.id}`)
        )
        results.push({
          id: segment.id,
          url: existingBlob?.url || '',
          duration: 0,
          characters: segment.text.length,
          status: 'skipped',
        })
        continue
      }

      try {
        console.log(`[ziggurat-audio] Generating segment ${segment.id}...`)
        const result = await generateSegment(segment)
        results.push({
          ...result,
          status: 'generated',
        })
        
        // Small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 500))
      } catch (error) {
        console.error(`[ziggurat-audio] Error generating segment ${segment.id}:`, error)
        results.push({
          id: segment.id,
          url: '',
          duration: 0,
          characters: segment.text.length,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error',
        })
      }
    }

    const generated = results.filter(r => r.status === 'generated').length
    const skipped = results.filter(r => r.status === 'skipped').length
    const errors = results.filter(r => r.status === 'error').length
    const totalCharacters = results.reduce((sum, r) => sum + r.characters, 0)

    return NextResponse.json({
      status: errors === 0 ? 'success' : 'partial',
      generated,
      skipped,
      errors,
      totalCharacters,
      estimatedCost: `~${Math.ceil(totalCharacters / 1000)} credits`,
      segments: results,
      nextStep: errors === 0 
        ? 'All segments generated. Nicolette can now run the ffmpeg assembly script.'
        : `${errors} segments failed. Retry with ?segment=XX to regenerate specific segments.`,
    })
  } catch (error) {
    console.error('[ziggurat-audio] Fatal error:', error)
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}
