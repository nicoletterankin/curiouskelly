import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// In-memory cache for translations
const translationCache = new Map<string, { data: unknown; timestamp: number }>()
const CACHE_TTL = 300000 // 5 minutes

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '1')
  const lang = searchParams.get('lang') || 'en'
  
  // English is the source - no translation needed
  if (lang === 'en') {
    return NextResponse.json({
      day,
      language: 'en',
      translation: null,
      fallback: false,
    })
  }
  
  const cacheKey = `${day}-${lang}`
  const cached = translationCache.get(cacheKey)
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return NextResponse.json(cached.data)
  }
  
  try {
    // First try active_translations (simpler, with audio URLs)
    const activeTranslations = await sql`
      SELECT phase, translated_text, audio_url
      FROM active_translations
      WHERE day_number = ${day} AND language = ${lang}
    `
    
    if (activeTranslations.length > 0) {
      const phases: Record<string, { text: string; audio_url?: string }> = {}
      for (const t of activeTranslations) {
        phases[t.phase] = { text: t.translated_text, audio_url: t.audio_url }
      }
      const result = {
        day,
        language: lang,
        translation: phases,
        hasAudio: activeTranslations.some((t: { audio_url: string | null }) => t.audio_url),
        fallback: false,
      }
      translationCache.set(cacheKey, { data: result, timestamp: Date.now() })
      return NextResponse.json(result)
    }
    
    // Fallback to lesson_translations table (uses lesson_id not day_of_year)
    const translations = await sql`
      SELECT 
        title,
        hook_script,
        story_script,
        wonder_script,
        action_script,
        wisdom_script,
        cultural_notes,
        local_examples
      FROM lesson_translations
      WHERE lesson_id = ${day.toString()} 
        AND language_code = ${lang}
        AND status = 'published'
      LIMIT 1
    `
    
    if (translations.length > 0) {
      const result = {
        day,
        language: lang,
        translation: translations[0],
        fallback: false,
      }
      translationCache.set(cacheKey, { data: result, timestamp: Date.now() })
      return NextResponse.json(result)
    }
    
    // No translation found - return fallback flag
    const result = {
      day,
      language: lang,
      translation: null,
      fallback: true,
    }
    translationCache.set(cacheKey, { data: result, timestamp: Date.now() })
    return NextResponse.json(result)
    
  } catch (error) {
    console.error('[translations] Error:', error)
    return NextResponse.json({
      day,
      language: lang,
      translation: null,
      fallback: true,
      error: 'Database error',
    })
  }
}

// POST: Submit a new translation (for admin/translator use)
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const {
      day,
      language,
      title,
      subtitle,
      hook_script,
      story_script,
      wonder_script,
      action_script,
      wisdom_script,
      quote,
      quote_author,
      status = 'draft',
    } = body
    
    if (!day || !language) {
      return NextResponse.json({ error: 'Missing day or language' }, { status: 400 })
    }
    
    // Upsert into lesson_translations (actual schema)
    await sql`
      INSERT INTO lesson_translations (
        translation_id,
        lesson_id,
        language_code,
        title,
        hook_script,
        story_script,
        wonder_script,
        action_script,
        wisdom_script,
        cultural_notes,
        local_examples,
        status,
        submitted_at
      ) VALUES (
        gen_random_uuid(),
        ${day.toString()},
        ${language},
        ${title},
        ${hook_script},
        ${story_script},
        ${wonder_script},
        ${action_script},
        ${wisdom_script},
        ${quote || ''},
        ${quote_author || ''},
        ${status},
        NOW()
      )
      ON CONFLICT (lesson_id, language_code) 
      DO UPDATE SET
        title = EXCLUDED.title,
        hook_script = EXCLUDED.hook_script,
        story_script = EXCLUDED.story_script,
        wonder_script = EXCLUDED.wonder_script,
        action_script = EXCLUDED.action_script,
        wisdom_script = EXCLUDED.wisdom_script,
        cultural_notes = EXCLUDED.cultural_notes,
        local_examples = EXCLUDED.local_examples,
        status = EXCLUDED.status
    `
    
    // Also update active_translations for quick lookup
    const phases = [
      { phase: 'hook', text: hook_script },
      { phase: 'story', text: story_script },
      { phase: 'wonder', text: wonder_script },
      { phase: 'action', text: action_script },
      { phase: 'wisdom', text: wisdom_script },
    ]
    
    for (const p of phases) {
      if (p.text) {
        await sql`
          INSERT INTO active_translations (day_number, phase, language, translated_text)
          VALUES (${day}, ${p.phase}, ${language}, ${p.text})
          ON CONFLICT (day_number, phase, language) DO UPDATE SET
            translated_text = ${p.text}
        `
      }
    }
    
    // Clear cache for this translation
    translationCache.delete(`${day}-${language}`)
    
    return NextResponse.json({ success: true, day, language })
    
  } catch (error) {
    console.error('[translations] POST error:', error)
    return NextResponse.json({ error: 'Failed to save translation' }, { status: 500 })
  }
}
