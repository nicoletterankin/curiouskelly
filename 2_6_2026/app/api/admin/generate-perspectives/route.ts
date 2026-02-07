import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { generateText } from 'ai'
import { openai } from '@ai-sdk/openai'

const AGE_GROUPS = ['kid', 'adult', 'senior'] as const
const ARCHETYPES = ['storyteller', 'scientist', 'explorer', 'rebel', 'mystic', 'macgyver'] as const
const LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'pt'] as const

const AGE_PROMPTS = {
  kid: 'for a curious 6-8 year old child. Use simple words, excitement, wonder, and playful language. Make it fun!',
  adult: 'for a curious adult learner. Use clear, engaging language with interesting details.',
  senior: 'for a wise elder. Use respectful, thoughtful language that connects to life experience and wisdom.',
}

const ARCHETYPE_PROMPTS = {
  storyteller: 'as a warm storyteller who weaves narratives and makes everything feel like an adventure tale.',
  scientist: 'as an enthusiastic scientist who loves explaining how things work with fascinating facts.',
  explorer: 'as an adventurous explorer who discovers wonders in everyday things.',
  rebel: 'as a playful rebel who questions everything and finds the surprising truth.',
  mystic: 'as a thoughtful mystic who sees deeper meaning and magic in nature.',
  macgyver: 'as a clever inventor who sees practical applications and hands-on experiments everywhere.',
}

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  es: 'Spanish',
  fr: 'French', 
  de: 'German',
  zh: 'Mandarin Chinese',
  pt: 'Portuguese',
}

export async function POST(request: NextRequest) {
  try {
    const { dayNumber, ageGroup, archetype, language } = await request.json()
    
    const sql = getSql()
    // Get base lesson
    const lessons = await sql`
      SELECT title, topic, theme, hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lessons WHERE day_of_year = ${dayNumber}
    `
    
    if (!lessons.length) {
      return NextResponse.json({ error: 'Lesson not found' }, { status: 404 })
    }
    
    const lesson = lessons[0]
    const age = ageGroup as keyof typeof AGE_PROMPTS
    const arch = archetype as keyof typeof ARCHETYPE_PROMPTS
    const lang = language || 'en'
    
    const languageInstruction = lang !== 'en' 
      ? `Respond ENTIRELY in ${LANGUAGE_NAMES[lang]}. All text must be in ${LANGUAGE_NAMES[lang]}.`
      : ''
    
    const prompt = `You are Kelly, a warm AI teacher. Adapt this lesson ${AGE_PROMPTS[age]} Speak ${ARCHETYPE_PROMPTS[arch]}

${languageInstruction}

Original lesson:
Title: ${lesson.title}
Topic: ${lesson.topic}
Theme: ${lesson.theme}

Original scripts:
Hook: ${lesson.hook_script}
Story: ${lesson.story_script}  
Wonder: ${lesson.wonder_script}
Action: ${lesson.action_script}
Wisdom: ${lesson.wisdom_script}

Create a personalized version with:
1. A catchy title (age and tone appropriate)
2. A subtitle (one line)
3. Adapted scripts for each phase (hook, story, wonder, action, wisdom) - keep them concise (1-3 sentences each)

Respond in JSON format:
{
  "title": "...",
  "hook_script": "...",
  "story_script": "...",
  "wonder_script": "...",
  "action_script": "...",
  "wisdom_script": "..."
}`

    const { text } = await generateText({
      model: openai('gpt-4o-mini'),
      prompt,
    })
    
    // Parse JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/)
    if (!jsonMatch) {
      return NextResponse.json({ error: 'Failed to parse AI response' }, { status: 500 })
    }
    
    const perspective = JSON.parse(jsonMatch[0])
    
    // Save to database
    await sql`
      INSERT INTO lesson_perspectives (
        day_number, age_group, archetype, language,
        title, topic, theme,
        hook_script, story_script, wonder_script, action_script, wisdom_script
      ) VALUES (
        ${dayNumber}, ${ageGroup}, ${archetype}, ${lang},
        ${perspective.title}, ${lesson.topic}, ${lesson.theme},
        ${perspective.hook_script}, ${perspective.story_script}, 
        ${perspective.wonder_script}, ${perspective.action_script}, ${perspective.wisdom_script}
      )
      ON CONFLICT (day_number, age_group, archetype, language) 
      DO UPDATE SET
        title = EXCLUDED.title,
        hook_script = EXCLUDED.hook_script,
        story_script = EXCLUDED.story_script,
        wonder_script = EXCLUDED.wonder_script,
        action_script = EXCLUDED.action_script,
        wisdom_script = EXCLUDED.wisdom_script,
        updated_at = NOW()
    `
    
    return NextResponse.json({ 
      success: true, 
      perspective: {
        day_number: dayNumber,
        age_group: ageGroup,
        archetype,
        language: lang,
        ...perspective
      }
    })
  } catch (error) {
    console.error('Generate perspective error:', error)
    return NextResponse.json({ error: 'Failed to generate perspective' }, { status: 500 })
  }
}

// Batch generate for a day
export async function PUT(request: NextRequest) {
  try {
    const { dayNumber, batchSize = 18 } = await request.json()
    
    const results: Array<{ age: string; archetype: string; lang: string; status: string }> = []
    let count = 0
    
    const sql = getSql()
    for (const age of AGE_GROUPS) {
      for (const arch of ARCHETYPES) {
        for (const lang of LANGUAGES) {
          if (count >= batchSize) break
          
          // Check if already exists
          const existing = await sql`
            SELECT id FROM lesson_perspectives 
            WHERE day_number = ${dayNumber} AND age_group = ${age} AND archetype = ${arch} AND language = ${lang}
          `
          
          if (existing.length > 0) {
            results.push({ age, archetype: arch, lang, status: 'exists' })
            continue
          }
          
          // Generate via POST
          const res = await fetch(request.url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dayNumber, ageGroup: age, archetype: arch, language: lang })
          })
          
          results.push({ 
            age, 
            archetype: arch, 
            lang, 
            status: res.ok ? 'generated' : 'failed' 
          })
          count++
          
          // Rate limit
          await new Promise(r => setTimeout(r, 500))
        }
        if (count >= batchSize) break
      }
      if (count >= batchSize) break
    }
    
    return NextResponse.json({ 
      success: true, 
      dayNumber,
      generated: count,
      results 
    })
  } catch (error) {
    console.error('Batch generate error:', error)
    return NextResponse.json({ error: 'Batch generation failed' }, { status: 500 })
  }
}

// Get status
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const dayNumber = searchParams.get('day')
  
  const sql = getSql()
  if (dayNumber) {
    const perspectives = await sql`
      SELECT day_number, age_group, archetype, language, title
      FROM lesson_perspectives WHERE day_number = ${parseInt(dayNumber)}
      ORDER BY age_group, archetype, language
    `
    return NextResponse.json({ dayNumber, count: perspectives.length, perspectives })
  }
  
  // Overall stats
  const stats = await sql`
    SELECT 
      COUNT(*)::int as total,
      COUNT(DISTINCT day_number)::int as days_covered,
      (SELECT COUNT(*) FROM lessons)::int as total_days
    FROM lesson_perspectives
  `
  
  return NextResponse.json(stats[0])
}
