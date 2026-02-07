import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * UNIFIED KELLY ASSET API
 * 
 * Smart fallback chain for video/audio/text:
 * 1. Exact match (day + age + archetype + language + phase)
 * 2. Same day/age/archetype, English (if non-English requested)
 * 3. Same day/age, default archetype (storyteller)
 * 4. Audio-only with Kelly image
 * 5. Text-only with TTS queued
 * 
 * This is THE endpoint for all Kelly content requests.
 */

interface AssetResponse {
  day: number
  phase: string
  type: 'video' | 'audio' | 'text'
  url: string | null
  audioUrl: string | null
  text: string
  title: string
  fallbackLevel: number // 1=exact, 2=language, 3=archetype, 4=audio, 5=text
  requested: {
    age: string
    archetype: string
    language: string
  }
  served: {
    age: string
    archetype: string
    language: string
  }
}

const PHASE_TO_SCRIPT: Record<string, string> = {
  hook: 'hook_script',
  story: 'story_script',
  wonder: 'wonder_script',
  action: 'action_script',
  wisdom: 'wisdom_script'
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '1')
  const phase = searchParams.get('phase') || 'hook'
  const age = searchParams.get('age') || 'adult'
  const archetype = searchParams.get('archetype') || 'storyteller'
  const language = searchParams.get('language') || 'en'
  
  const requested = { age, archetype, language }
  
  try {
    // LEVEL 1: Exact match in kelly_lesson_assets
    const exactMatch = await sql`
      SELECT video_url, audio_url 
      FROM kelly_lesson_assets 
      WHERE day_number = ${day} 
        AND phase = ${phase}
        AND age_group = ${age}
        AND archetype = ${archetype}
        AND language = ${language}
        AND (video_url IS NOT NULL OR audio_url IS NOT NULL)
      LIMIT 1
    `
    
    if (exactMatch.length > 0 && exactMatch[0].video_url) {
      const text = await getScriptText(day, phase, language)
      const title = await getLessonTitle(day, language)
      return NextResponse.json({
        day, phase, type: 'video',
        url: exactMatch[0].video_url,
        audioUrl: exactMatch[0].audio_url,
        text, title,
        fallbackLevel: 1,
        requested,
        served: { age, archetype, language }
      } as AssetResponse)
    }
    
    // LEVEL 2: Same params but English (for non-English requests)
    if (language !== 'en') {
      const englishMatch = await sql`
        SELECT video_url, audio_url 
        FROM kelly_lesson_assets 
        WHERE day_number = ${day} 
          AND phase = ${phase}
          AND age_group = ${age}
          AND archetype = ${archetype}
          AND language = 'en'
          AND video_url IS NOT NULL
        LIMIT 1
      `
      
      if (englishMatch.length > 0 && englishMatch[0].video_url) {
        const text = await getScriptText(day, phase, language) // Still serve translated text
        const title = await getLessonTitle(day, language)
        return NextResponse.json({
          day, phase, type: 'video',
          url: englishMatch[0].video_url,
          audioUrl: englishMatch[0].audio_url,
          text, title,
          fallbackLevel: 2,
          requested,
          served: { age, archetype, language: 'en' }
        } as AssetResponse)
      }
    }
    
    // LEVEL 3: Same day/age, default archetype (storyteller)
    if (archetype !== 'storyteller') {
      const storytellerMatch = await sql`
        SELECT video_url, audio_url 
        FROM kelly_lesson_assets 
        WHERE day_number = ${day} 
          AND phase = ${phase}
          AND age_group = ${age}
          AND archetype = 'storyteller'
          AND video_url IS NOT NULL
        LIMIT 1
      `
      
      if (storytellerMatch.length > 0 && storytellerMatch[0].video_url) {
        const text = await getScriptText(day, phase, language)
        const title = await getLessonTitle(day, language)
        return NextResponse.json({
          day, phase, type: 'video',
          url: storytellerMatch[0].video_url,
          audioUrl: storytellerMatch[0].audio_url,
          text, title,
          fallbackLevel: 3,
          requested,
          served: { age, archetype: 'storyteller', language: language === 'en' ? 'en' : 'en' }
        } as AssetResponse)
      }
    }
    
    // LEVEL 4: Any video for this day/phase (ignore age/archetype)
    const anyVideo = await sql`
      SELECT video_url, audio_url, age_group, archetype, language
      FROM kelly_lesson_assets 
      WHERE day_number = ${day} 
        AND phase = ${phase}
        AND video_url IS NOT NULL
      LIMIT 1
    `
    
    if (anyVideo.length > 0 && anyVideo[0].video_url) {
      const text = await getScriptText(day, phase, language)
      const title = await getLessonTitle(day, language)
      return NextResponse.json({
        day, phase, type: 'video',
        url: anyVideo[0].video_url,
        audioUrl: anyVideo[0].audio_url,
        text, title,
        fallbackLevel: 4,
        requested,
        served: { 
          age: anyVideo[0].age_group, 
          archetype: anyVideo[0].archetype,
          language: anyVideo[0].language 
        }
      } as AssetResponse)
    }
    
    // LEVEL 5: Audio-only (TTS)
    const audioOnly = await sql`
      SELECT audio_url, language
      FROM lesson_audio 
      WHERE day_number = ${day}
        AND (language = ${language} OR language = 'en')
      ORDER BY CASE WHEN language = ${language} THEN 0 ELSE 1 END
      LIMIT 1
    `
    
    if (audioOnly.length > 0 && audioOnly[0].audio_url) {
      const text = await getScriptText(day, phase, language)
      const title = await getLessonTitle(day, language)
      return NextResponse.json({
        day, phase, type: 'audio',
        url: null,
        audioUrl: audioOnly[0].audio_url,
        text, title,
        fallbackLevel: 5,
        requested,
        served: { age, archetype, language: audioOnly[0].language }
      } as AssetResponse)
    }
    
    // LEVEL 6: Text-only fallback
    const text = await getScriptText(day, phase, language)
    const title = await getLessonTitle(day, language)
    
    return NextResponse.json({
      day, phase, type: 'text',
      url: null,
      audioUrl: null,
      text,
      title,
      fallbackLevel: 6,
      requested,
      served: { age, archetype, language }
    } as AssetResponse)
    
  } catch (error) {
    console.error('Asset API error:', error)
    return NextResponse.json({ error: 'Failed to fetch asset' }, { status: 500 })
  }
}

async function getScriptText(day: number, phase: string, language: string): Promise<string> {
  const scriptCol = PHASE_TO_SCRIPT[phase] || 'hook_script'
  
  // Try translation first
  if (language !== 'en') {
    const translation = await sql`
      SELECT ${sql(scriptCol)} as script
      FROM lesson_translations
      WHERE lesson_id = ${day.toString()}
        AND language_code = ${language}
      LIMIT 1
    `.catch(() => [])
    
    if (translation.length > 0 && translation[0].script) {
      return translation[0].script
    }
  }
  
  // Fall back to English
  const lesson = await sql`
    SELECT ${sql(scriptCol)} as script
    FROM lessons
    WHERE day_of_year = ${day}
    LIMIT 1
  `.catch(() => [])
  
  return lesson[0]?.script || ''
}

async function getLessonTitle(day: number, language: string): Promise<string> {
  // Try translation first
  if (language !== 'en') {
    const translation = await sql`
      SELECT title
      FROM lesson_translations
      WHERE lesson_id = ${day.toString()}
        AND language_code = ${language}
      LIMIT 1
    `.catch(() => [])
    
    if (translation.length > 0 && translation[0].title) {
      return translation[0].title
    }
  }
  
  // Fall back to English
  const lesson = await sql`
    SELECT title
    FROM lessons
    WHERE day_of_year = ${day}
    LIMIT 1
  `.catch(() => [])
  
  return lesson[0]?.title || "Today's Lesson"
}
