import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { generateText } from 'ai'
import { openai } from '@ai-sdk/openai'

/**
 * BATCH TRANSLATION GENERATOR
 * 
 * Uses AI to translate lesson scripts into 5 languages:
 * ES (Spanish), FR (French), DE (German), PT (Portuguese), ZH (Chinese)
 * 
 * POST /api/translations/batch-generate
 * Body: { dayStart: 1, dayEnd: 30, languages?: string[] }
 * 
 * GET /api/translations/batch-generate
 * Returns coverage stats
 */

const LANGUAGES = [
  { code: 'es', name: 'Spanish', nativeName: 'Español' },
  { code: 'fr', name: 'French', nativeName: 'Français' },
  { code: 'de', name: 'German', nativeName: 'Deutsch' },
  { code: 'pt', name: 'Portuguese', nativeName: 'Português' },
  { code: 'zh', name: 'Chinese (Simplified)', nativeName: '简体中文' },
]

async function translateLesson(
  lesson: {
    day_of_year: number
    title: string
    hook_script: string
    story_script: string
    wonder_script: string
    action_script: string
    wisdom_script: string
  },
  targetLang: { code: string; name: string; nativeName: string }
): Promise<{
  title: string
  hook_script: string
  story_script: string
  wonder_script: string
  action_script: string
  wisdom_script: string
} | null> {
  try {
    const prompt = `You are a professional translator for an educational app for children and families.
Translate the following lesson content from English to ${targetLang.name} (${targetLang.nativeName}).

IMPORTANT RULES:
1. Keep the same tone - warm, curious, engaging for kids aged 5-15
2. Adapt cultural references if needed (but keep the core concept)
3. Keep sentences short and clear
4. DO NOT add or remove content - just translate
5. Return ONLY valid JSON, no markdown or explanation

Input lesson (Day ${lesson.day_of_year}):
{
  "title": ${JSON.stringify(lesson.title)},
  "hook_script": ${JSON.stringify(lesson.hook_script)},
  "story_script": ${JSON.stringify(lesson.story_script)},
  "wonder_script": ${JSON.stringify(lesson.wonder_script)},
  "action_script": ${JSON.stringify(lesson.action_script)},
  "wisdom_script": ${JSON.stringify(lesson.wisdom_script)}
}

Return the translation as JSON with these exact keys:
{
  "title": "translated title",
  "hook_script": "translated hook",
  "story_script": "translated story",
  "wonder_script": "translated wonder fact",
  "action_script": "translated action/activity",
  "wisdom_script": "translated wisdom/reflection"
}`

    const { text } = await generateText({
      model: openai('gpt-4o-mini'),
      prompt,
      temperature: 0.3,
    })

    // Parse the JSON response
    const cleaned = text.replace(/```json\n?|\n?```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    
    return {
      title: parsed.title || lesson.title,
      hook_script: parsed.hook_script || lesson.hook_script,
      story_script: parsed.story_script || lesson.story_script,
      wonder_script: parsed.wonder_script || lesson.wonder_script,
      action_script: parsed.action_script || lesson.action_script,
      wisdom_script: parsed.wisdom_script || lesson.wisdom_script,
    }
  } catch (e) {
    console.error(`[translations] Failed to translate day ${lesson.day_of_year} to ${targetLang.code}:`, e)
    return null
  }
}

export async function GET() {
  // Get coverage stats from lesson_translations table
  // Actual schema: translation_id, lesson_id, language_code, title, hook_script, story_script, 
  //                wonder_script, action_script, wisdom_script, status, submitted_at, etc.
  const coverage = await sql`
    SELECT 
      language_code,
      COUNT(*) as count,
      COUNT(*) FILTER (WHERE status = 'published') as published
    FROM lesson_translations
    WHERE language_code != 'en'
    GROUP BY language_code
    ORDER BY count DESC
  `
  
  // Also check active_translations table (simpler schema for production use)
  const activeCount = await sql`
    SELECT language, COUNT(*) as count
    FROM active_translations
    GROUP BY language
  `
  
  const missingByLang: Record<string, number[]> = {}
  
  for (const lang of LANGUAGES) {
    // Check which days have translations
    const existing = await sql`
      SELECT DISTINCT 
        CASE 
          WHEN lesson_id ~ '^[0-9]+$' THEN lesson_id::int 
          ELSE 0 
        END as day_num
      FROM lesson_translations 
      WHERE language_code = ${lang.code}
    `
    const existingDays = new Set(existing.map((r: { day_num: number }) => r.day_num))
    const missing: number[] = []
    for (let d = 1; d <= 365; d++) {
      if (!existingDays.has(d)) missing.push(d)
    }
    if (missing.length > 0) {
      missingByLang[lang.code] = missing.slice(0, 10) // Show first 10 missing
    }
  }
  
  return NextResponse.json({
    totalLessons: 365,
    targetLanguages: LANGUAGES.map(l => l.code),
    coverage: coverage.reduce((acc: Record<string, number>, row: { language_code: string; count: string }) => {
      acc[row.language_code] = Number(row.count)
      return acc
    }, {}),
    activeTranslations: activeCount.reduce((acc: Record<string, number>, row: { language: string; count: string }) => {
      acc[row.language] = Number(row.count)
      return acc
    }, {}),
    totalNeeded: 365 * 5,
    missingByLanguage: missingByLang,
    endpoint: 'POST with { dayStart: 1, dayEnd: 30, languages: ["es", "fr"] }',
  })
}

export async function POST(request: NextRequest) {
  const { dayStart = 1, dayEnd = 30, languages = ['es', 'fr', 'de', 'pt', 'zh'] } = await request.json()
  
  const targetLangs = LANGUAGES.filter(l => languages.includes(l.code))
  
  if (targetLangs.length === 0) {
    return NextResponse.json({ error: 'No valid languages specified' }, { status: 400 })
  }
  
  // Get lessons for the specified range
  const lessons = await sql`
    SELECT day_of_year, title, hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons
    WHERE day_of_year >= ${dayStart} AND day_of_year <= ${dayEnd}
    ORDER BY day_of_year
  `
  
  if (lessons.length === 0) {
    return NextResponse.json({ error: 'No lessons found in range' }, { status: 404 })
  }
  
  const results: { 
    created: string[]
    skipped: string[]
    failed: string[]
  } = { created: [], skipped: [], failed: [] }
  
  for (const lesson of lessons) {
    for (const lang of targetLangs) {
      const key = `day${lesson.day_of_year}-${lang.code}`
      const lessonId = lesson.day_of_year.toString()
      
      // Check if translation already exists in lesson_translations
      const existing = await sql`
        SELECT translation_id FROM lesson_translations
        WHERE lesson_id = ${lessonId} AND language_code = ${lang.code}
        LIMIT 1
      `
      
      if (existing.length > 0) {
        results.skipped.push(key)
        continue
      }
      
      // Generate translation
      const translation = await translateLesson(lesson, lang)
      
      if (!translation) {
        results.failed.push(key)
        continue
      }
      
      // Save to lesson_translations table (actual schema)
      await sql`
        INSERT INTO lesson_translations (
          translation_id, lesson_id, language_code, title,
          hook_script, story_script, wonder_script, action_script, wisdom_script,
          status, submitted_at, word_count
        ) VALUES (
          gen_random_uuid(),
          ${lessonId},
          ${lang.code},
          ${translation.title},
          ${translation.hook_script},
          ${translation.story_script},
          ${translation.wonder_script},
          ${translation.action_script},
          ${translation.wisdom_script},
          'published',
          NOW(),
          ${(translation.hook_script + translation.story_script + translation.wonder_script + translation.action_script + translation.wisdom_script).split(/\s+/).length}
        )
      `
      
      // Also insert into active_translations for quick lookup
      for (const phase of ['hook', 'story', 'wonder', 'action', 'wisdom']) {
        const scriptKey = `${phase}_script` as keyof typeof translation
        const scriptText = translation[scriptKey]
        if (scriptText) {
          await sql`
            INSERT INTO active_translations (day_number, phase, language, translated_text)
            VALUES (${lesson.day_of_year}, ${phase}, ${lang.code}, ${scriptText})
            ON CONFLICT (day_number, phase, language) DO UPDATE SET
              translated_text = ${scriptText}
          `
        }
      }
      
      results.created.push(key)
      
      // Rate limit - small delay between API calls
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
  
  return NextResponse.json({
    status: 'complete',
    dayRange: { start: dayStart, end: dayEnd },
    languages: targetLangs.map(l => l.code),
    results,
    summary: {
      created: results.created.length,
      skipped: results.skipped.length,
      failed: results.failed.length,
    },
  })
}
