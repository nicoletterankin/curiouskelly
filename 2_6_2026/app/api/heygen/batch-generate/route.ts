import { NextRequest, NextResponse } from 'next/server'
import { generateLessonVideos, type Archetype, type AgeCategory } from '@/lib/heygen-client'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const {
      dayOfYear,
      age,
      archetype,
      scripts,
      audioUrls,
    } = body as {
      dayOfYear: number
      age: number | AgeCategory
      archetype: Archetype
      scripts: {
        hook: string
        story: string
        wonder: string
        action: string
        wisdom: string
      }
      audioUrls?: {
        hook?: string
        story?: string
        wonder?: string
        action?: string
        wisdom?: string
      }
    }
    
    // Validate required fields
    if (!dayOfYear || !age || !archetype || !scripts) {
      return NextResponse.json(
        { error: 'Missing required fields: dayOfYear, age, archetype, scripts' },
        { status: 400 }
      )
    }
    
    // Validate scripts object has all phases
    const phases = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
    for (const phase of phases) {
      if (!scripts[phase] && !audioUrls?.[phase]) {
        return NextResponse.json(
          { error: `Missing script or audioUrl for phase: ${phase}` },
          { status: 400 }
        )
      }
    }
    
    // Generate all 5 videos
    const results = await generateLessonVideos({
      dayOfYear,
      age,
      archetype,
      scripts,
      audioUrls,
    })
    
    return NextResponse.json({
      success: true,
      data: {
        dayOfYear,
        age,
        archetype,
        videos: results,
        creditsUsed: 5, // Estimate
      },
    })
  } catch (error) {
    console.error('[HeyGen] Batch generation error:', error)
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Batch generation failed',
        success: false,
      },
      { status: 500 }
    )
  }
}
