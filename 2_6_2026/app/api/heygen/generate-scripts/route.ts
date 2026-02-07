import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { 
  generateScript, 
  generateUserScripts, 
  generateAllScriptsForLesson,
  type LessonData,
  type Phase,
} from '@/lib/script-generator'
import type { Archetype, AgeCategory } from '@/lib/heygen-client'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const {
      dayOfYear,
      mode = 'user', // 'user' | 'single' | 'all'
      // For 'user' mode
      userAge,
      userArchetype,
      // For 'single' mode
      phase,
      archetype,
      ageCategory,
    } = body as {
      dayOfYear: number
      mode?: 'user' | 'single' | 'all'
      userAge?: number
      userArchetype?: Archetype
      phase?: Phase
      archetype?: Archetype
      ageCategory?: AgeCategory
    }
    
    if (!dayOfYear) {
      return NextResponse.json(
        { error: 'dayOfYear is required' },
        { status: 400 }
      )
    }
    
    const sql = getSql()
    // Fetch lesson data from database
    const lessons = await sql`
      SELECT 
        day_of_year,
        topic,
        theme,
        kelly_age,
        hook_script,
        story_script,
        wonder_script,
        action_script,
        wisdom_script
      FROM lessons 
      WHERE day_of_year = ${dayOfYear}
      LIMIT 1
    `
    
    if (lessons.length === 0) {
      return NextResponse.json(
        { error: `Lesson not found for day ${dayOfYear}` },
        { status: 404 }
      )
    }
    
    const row = lessons[0]
    
    // Parse the lesson data
    // Note: Scripts are stored as JSON strings in the database
    const lessonData: LessonData = {
      dayOfYear: row.day_of_year,
      topic: row.topic,
      theme: row.theme,
      kellyAge: row.kelly_age,
      phases: {
        hook: { question: row.hook_script || '' },
        story: { narrative: row.story_script || '' },
        wonder: { exploration: row.wonder_script || '' },
        action: { activity: row.action_script || '' },
        wisdom: { 
          quote: '', // TODO: Parse from wisdom_script
          quoteAuthor: '',
          reflection: row.wisdom_script || '',
        },
      },
    }
    
    let scripts
    
    switch (mode) {
      case 'user':
        if (!userAge || !userArchetype) {
          return NextResponse.json(
            { error: 'userAge and userArchetype required for user mode' },
            { status: 400 }
          )
        }
        scripts = generateUserScripts(lessonData, userAge, userArchetype)
        break
        
      case 'single':
        if (!phase || !archetype || !ageCategory) {
          return NextResponse.json(
            { error: 'phase, archetype, and ageCategory required for single mode' },
            { status: 400 }
          )
        }
        scripts = [generateScript(lessonData, phase, archetype, ageCategory)]
        break
        
      case 'all':
        scripts = generateAllScriptsForLesson(lessonData)
        break
        
      default:
        return NextResponse.json(
          { error: 'Invalid mode. Use: user, single, or all' },
          { status: 400 }
        )
    }
    
    return NextResponse.json({
      success: true,
      data: {
        dayOfYear,
        topic: lessonData.topic,
        theme: lessonData.theme,
        kellyAge: lessonData.kellyAge,
        mode,
        scriptCount: Array.isArray(scripts) ? scripts.length : 5,
        scripts,
      },
    })
  } catch (error) {
    console.error('[Scripts] Generation error:', error)
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Script generation failed',
        success: false,
      },
      { status: 500 }
    )
  }
}
