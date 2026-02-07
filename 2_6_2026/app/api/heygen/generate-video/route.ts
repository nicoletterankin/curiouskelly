import { NextRequest, NextResponse } from 'next/server'
import { generateKellyVideo, type Archetype, type AgeCategory } from '@/lib/heygen-client'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const {
      age,
      archetype,
      script,
      audioUrl,
      title,
      test = false,
    } = body as {
      age: number | AgeCategory
      archetype: Archetype
      script: string
      audioUrl?: string
      title?: string
      test?: boolean
    }
    
    // Validate required fields
    if (!age || !archetype || (!script && !audioUrl)) {
      return NextResponse.json(
        { error: 'Missing required fields: age, archetype, and either script or audioUrl' },
        { status: 400 }
      )
    }
    
    // Validate archetype
    const validArchetypes = [
      'scientist', 'explorer', 'rebel', 'architect',
      'diplomat', 'empath', 'macgyver', 'mystic',
      'provider', 'storyteller', 'strategist', 'survivor',
    ]
    
    if (!validArchetypes.includes(archetype)) {
      return NextResponse.json(
        { error: `Invalid archetype. Must be one of: ${validArchetypes.join(', ')}` },
        { status: 400 }
      )
    }
    
    // Validate age
    const validAgeCategories = ['kid', 'adult', 'senior']
    if (typeof age === 'string' && !validAgeCategories.includes(age)) {
      return NextResponse.json(
        { error: `Invalid age category. Must be one of: ${validAgeCategories.join(', ')}` },
        { status: 400 }
      )
    }
    
    // Generate video
    const result = await generateKellyVideo({
      age,
      archetype,
      script,
      audioUrl,
      title,
      test,
    })
    
    return NextResponse.json({
      success: true,
      data: result,
    })
  } catch (error) {
    console.error('[HeyGen] Video generation error:', error)
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Video generation failed',
        success: false,
      },
      { status: 500 }
    )
  }
}
