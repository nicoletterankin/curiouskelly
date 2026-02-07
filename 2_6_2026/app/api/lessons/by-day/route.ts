import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { formatFullDate, getDayOfYear } from '@/lib/date-utils'
import { lessonCache, CACHE_TTL } from '@/lib/cache'
import { ARCHETYPE_PERSONALITIES, SUPPORTED_LANGUAGES, type Archetype } from '@/lib/kelly-assets'

// Force dynamic rendering - no edge caching
// Build: 2026-02-05T14:00:00Z - Connected to soft-block-64917198 - lesson_perspectives ENABLED (39K rows)
export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const day = searchParams.get('day')
    const timezone = searchParams.get('timezone') || 'UTC'
    const age = searchParams.get('age')
    const archetype = searchParams.get('archetype')
    const rawLanguage = searchParams.get('language') || 'en'
    
    // Validate language code - use canonical list from kelly-assets
    // Extended languages for translations: ja, ko, ar, hi, it, ru (added beyond video generation set)
    const EXTENDED_LANGUAGES = [...SUPPORTED_LANGUAGES, 'ja', 'ko', 'ar', 'hi', 'it', 'ru'] as const
    const language = EXTENDED_LANGUAGES.includes(rawLanguage as typeof EXTENDED_LANGUAGES[number]) ? rawLanguage : 'en'

    if (!day) {
      return NextResponse.json(
        { error: 'day parameter is required' },
        { status: 400 }
      )
    }

    const dayOfYear = parseInt(day, 10)
    if (isNaN(dayOfYear) || dayOfYear < 1 || dayOfYear > 366) {
      return NextResponse.json(
        { error: 'Invalid day of year' },
        { status: 400 }
      )
    }

    // DISABLED cache - language changes need fresh data every time
    // const cached = lessonCache.get(dayOfYear)
    // Cache bypass: Always fetch fresh when language changes

    // First, check for translated content if language is not English
    // Actual schema: lesson_translations has lesson_id (text), language_code (text), status (text)
    // Also check active_translations which has day_number (int), phase, language, translated_text, audio_url
    let translatedContent: Record<string, unknown> | null = null
    if (language !== 'en') {
      try {
        // First try active_translations (has audio URLs)
        const activeT = await sql`
          SELECT phase, translated_text, audio_url
          FROM active_translations
          WHERE day_number = ${dayOfYear} AND language = ${language}
        `
        
        if (activeT.length > 0) {
          // Build translated content from active_translations
          const phases: Record<string, string> = {}
          const audioUrls: Record<string, string> = {}
          for (const t of activeT) {
            const record = t as { phase: string; translated_text: string; audio_url: string | null }
            phases[record.phase] = record.translated_text
            if (record.audio_url) audioUrls[record.phase] = record.audio_url
          }
          translatedContent = {
            hook_script: phases['hook'],
            story_script: phases['story'],
            wonder_script: phases['wonder'],
            action_script: phases['action'],
            wisdom_script: phases['wisdom'],
            audio_urls: audioUrls,
          }
          // Active translations found for this language
        } else {
          // Fallback: Query lesson_translations by day_of_year
          const translations = await sql`
            SELECT id, day_of_year, language_code, title, hook_script, story_script, 
                   wonder_script, action_script, wisdom_script, cultural_notes, local_examples, status
            FROM lesson_translations 
            WHERE day_of_year = ${dayOfYear} 
              AND language_code = ${language} 
              AND status = 'published' 
            LIMIT 1
          `
          if (translations.length > 0) {
            translatedContent = translations[0] as Record<string, unknown>
          }
          // No translation found - will fall back to English
        }
      } catch (e) {
        // Gracefully fall back to English if translation lookup fails
        console.warn('[lessons/by-day] Translation lookup failed, using English:', e instanceof Error ? e.message : e)
      }
    }

    // Fetch base lesson by day_of_year
    let lessons: Record<string, unknown>[] = []
    try {
      lessons = await sql`SELECT * FROM lessons WHERE day_of_year = ${dayOfYear} LIMIT 1`
      if (lessons.length > 0) {
        console.log(`[v0] Day ${dayOfYear}: "${(lessons[0] as {title?: string}).title}" loaded OK`)
      } else {
        console.log(`[v0] Day ${dayOfYear}: No lesson found in DB`)
      }
    } catch (dbError) {
      console.warn('[lessons/by-day] DB error, using fallback')
      return NextResponse.json({
        lesson: {
          id: 0,
          day_of_year: dayOfYear,
          title: 'Loading...',
          summary: 'Please wait while we load today\'s lesson.',
          hook_script: '',
          story_script: '',
          wonder_script: '',
          action_script: '',
          wisdom_script: '',
        },
        perspective: null,
        displayDate: new Date().toLocaleDateString(),
        dayOfYear,
        language: 'en',
        error: 'Database temporarily unavailable'
      }, {
        headers: { 'Cache-Control': 'no-store' }
      })
    }

    if (lessons.length === 0) {
      return NextResponse.json(
        { error: 'Lesson not found for this day' },
        { status: 404 }
      )
    }

    const baseLesson = lessons[0] as Record<string, unknown>
    
    // Merge translated content if available
    // Use nullish coalescing (??) instead of OR (||) to preserve empty strings
    // This fixes the bug where empty string "" was treated as falsy and incorrectly fell back to English
    const lesson = translatedContent ? {
      ...baseLesson,
      title: translatedContent.title ?? baseLesson.title,
      subtitle: translatedContent.subtitle ?? baseLesson.subtitle,
      hook_script: translatedContent.hook_script ?? baseLesson.hook_script,
      story_script: translatedContent.story_script ?? baseLesson.story_script,
      wonder_script: translatedContent.wonder_script ?? baseLesson.wonder_script,
      action_script: translatedContent.action_script ?? baseLesson.action_script,
      wisdom_script: translatedContent.wisdom_script ?? baseLesson.wisdom_script,
      quote: translatedContent.quote ?? baseLesson.quote,
      quote_author: translatedContent.quote_author ?? baseLesson.quote_author,
      _translated: true,
      _language: language,
    } : {
      ...baseLesson,
      _translated: false,
      _language: 'en',
    }
    
    // ARCHETYPE TRANSFORMATION - Track personality but DON'T modify scripts
    // Greeting/phrases are displayed separately in the UI, not concatenated to content
    // This preserves the raw hook_script for True/False voting
    const arch = (archetype || 'explorer') as Archetype
    const personality = ARCHETYPE_PERSONALITIES[arch]
    
    if (personality && lesson) {
      // Mark archetype applied (UI will display greeting separately)
      lesson._archetype = arch
      lesson._personality = personality.label
      lesson._greeting = personality.greeting // Pass greeting separately for UI
      lesson._phrases = personality.phrases // Pass phrases for story intro
    }
    
    // Lesson merged with translations (if available)
    
    // Cache the lesson (disabled)
    // lessonCache.set(dayOfYear, { data: lesson, timestamp: Date.now() })

    // lesson_perspectives: VERIFIED EXISTS with 39K rows - personalized scripts by age/archetype
    // Schema: day_number, age_group, archetype, language, title, topic, theme, hook_script, story_script, etc.
    // NOTE: subtitle column does NOT exist in this table - don't query it
    let perspective: Record<string, unknown> | null = null
    try {
      const ageCategory = age || 'adult'
      const arch = archetype || 'storyteller'
      
      const perspectives = await sql`
        SELECT title, topic, theme, hook_script, story_script, wonder_script, action_script, wisdom_script
        FROM lesson_perspectives
        WHERE day_number = ${dayOfYear}
          AND age_group = ${ageCategory}
          AND archetype = ${arch}
          AND language = ${language}
        LIMIT 1
      `
      
      if (perspectives.length > 0) {
        perspective = perspectives[0] as Record<string, unknown>
      }
    } catch (e) {
      console.warn('[lessons/by-day] lesson_perspectives lookup failed:', e instanceof Error ? e.message : e)
    }

    // Fetch kelly_scripts variant scripts (kid/adult/elder) - highest priority source
    // Maps numeric age to variant: kid (2-10), adult (18-64), elder (65+)
    let kellyVariantScripts: Record<string, string> = {}
    let kellyVariantSource: string | null = null
    try {
      const ageNum = age ? parseInt(String(age), 10) : 30
      const variant = ageNum <= 10 ? 'kid' : ageNum >= 65 ? 'elder' : 'adult'
      
      const variantRows = await sql`
        SELECT phase, script_text
        FROM kelly_scripts
        WHERE day_number = ${dayOfYear}
          AND variant = ${variant}
          AND script_text IS NOT NULL
        ORDER BY phase
      `
      
      for (const r of variantRows) {
        const record = r as { phase: string; script_text: string }
        if (record.script_text) {
          kellyVariantScripts[record.phase] = record.script_text
        }
      }
      
      if (Object.keys(kellyVariantScripts).length > 0) {
        kellyVariantSource = variant
      }
    } catch (e) {
      console.warn('[lessons/by-day] kelly_scripts variant lookup failed:', e instanceof Error ? e.message : e)
    }

    // Fetch kelly_lesson_assets script_text per phase for this variant (fallback if kelly_scripts not available)
    // These are the most specific word-by-word scripts Kelly speaks for each age/archetype/language combo
    let kellyScripts: Record<string, string> = {}
    try {
      const ageCategory = age || 'adult'
      const arch = archetype || 'storyteller'
      
      const assets = await sql`
        SELECT phase, script_text
        FROM kelly_lesson_assets
        WHERE day_number = ${dayOfYear}
          AND age_group = ${ageCategory}
          AND (archetype = ${arch} OR archetype IS NULL)
          AND language = ${language}
          AND script_text IS NOT NULL
        ORDER BY CASE WHEN archetype = ${arch} THEN 0 ELSE 1 END
      `
      
      for (const a of assets) {
        const record = a as { phase: string; script_text: string }
        // Only keep the first (most specific) match per phase
        if (!kellyScripts[record.phase] && record.script_text) {
          kellyScripts[record.phase] = record.script_text
        }
      }
    } catch (e) {
      console.warn('[lessons/by-day] kelly_lesson_assets script lookup failed:', e instanceof Error ? e.message : e)
    }

    // Merge: kelly_scripts (variant-specific) takes priority over kelly_lesson_assets
    const mergedKellyScripts = {
      ...kellyScripts,
      ...kellyVariantScripts,
    }

    // Fetch HeyGen videos for this day/age/archetype
    // Table: heygen_videos - NO language column! Columns: day_of_year, phase, age_category, archetype, video_url, status
    let heygenVideos: Record<string, string> = {}
    try {
      const ageCategory = age || 'adult'
      const arch = archetype || 'storyteller'
      
      const videos = await sql`
        SELECT phase, video_url
        FROM heygen_videos
        WHERE day_of_year = ${dayOfYear}
          AND age_category = ${ageCategory}
          AND archetype = ${arch}
          AND status = 'completed'
          AND video_url IS NOT NULL
        ORDER BY created_at DESC
      `
      
      for (const v of videos) {
        const record = v as { phase: string; video_url: string }
        if (!heygenVideos[record.phase]) {
          heygenVideos[record.phase] = record.video_url
        }
      }
    } catch (e) {
      console.warn('[lessons/by-day] HeyGen video lookup failed:', e instanceof Error ? e.message : e)
    }

    // Format display date for current year
    const now = new Date()
    const year = now.getFullYear()
    // Create a date for this day of year
    const lessonDate = new Date(year, 0, dayOfYear)
    const displayDate = formatFullDate(lessonDate, timezone, 'en-US')

    return NextResponse.json({
      lesson: {
        ...lesson,
        // Hook Phase Cinematic Flow:
        // 1. VOTEABLE FACT: hook_fact - A TRUE statement learner votes on ("Is this true?")
        //    MUST be a declarative statement, NEVER a question.
        //    Fallback chain: hook_fact -> wonder_script (always a factual statement)
        // 2. ENGAGEMENT QUESTION: hook_script - Kelly's curiosity spark ("Have you ever wondered...")
        // 3. TEACHING MOMENT: wonder_script - The full explanation revealed after voting
        hook_fact: lesson.hook_fact || lesson.wonder_script || null, // Fallback: derive fact from wonder_script
        hook_question: lesson.hook_script || null, // Kelly's engagement question (shown after vote)
        hook_correct_answer: lesson.hook_correct_answer ?? true, // Read from DB - mix of true/false for real engagement
        teaching_moment: lesson.wonder_script || null, // Full explanation revealed after voting
      },
      perspective,
      displayDate,
      dayOfYear,
      language: lesson._language || language,
      hasTranslation: !!translatedContent,
      // Kelly's word-by-word scripts per phase for this specific variant (age/archetype/language)
      // Precedence: kelly_scripts (variant) > kelly_lesson_assets > perspective scripts > base lesson scripts
      kellyScripts: Object.keys(mergedKellyScripts).length > 0 ? mergedKellyScripts : null,
      hasKellyScripts: Object.keys(mergedKellyScripts).length > 0,
      kellyVariantSource,
      videos: heygenVideos,
      hasVideos: Object.keys(heygenVideos).length > 0,
      _ts: Date.now(),
      _build: 'V12-word-by-word-scripts', // Kelly word-by-word scripts per variant for teleprompter
    }, {
      headers: {
        'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      }
    })
  } catch (error) {
    console.error('Error fetching lesson:', error)
    // Return minimal lesson structure to prevent app crash - NO cached fallback
    const dayParam = new URL(request.url).searchParams.get('day')
    const dayNum = dayParam ? parseInt(dayParam, 10) : 1
    return NextResponse.json({
      lesson: {
        id: 0,
        day_of_year: dayNum,
        title: 'Loading...',
        summary: 'Please wait while we load today\'s lesson.',
        hook_script: '',
        story_script: '',
        wonder_script: '',
        action_script: '',
        wisdom_script: '',
      },
      perspective: null,
      displayDate: new Date().toLocaleDateString(),
      dayOfYear: dayNum,
      error: 'Database temporarily unavailable'
    }, {
      headers: { 'Cache-Control': 'no-store' }
    })
  }
}
