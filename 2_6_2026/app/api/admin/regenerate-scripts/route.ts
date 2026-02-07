/**
 * ADMIN: Regenerate phase scripts for corrected curriculum topics
 * 
 * POST /api/admin/regenerate-scripts
 * Body: { day: 37 }
 * 
 * Reads the corrected title + learning_objective from the curriculum_lookup,
 * generates 5 phase scripts (hook, story, wonder, action, wisdom) + metadata
 * matching the pattern of Days 1-60, then updates:
 *   1. lessons table (hook_script, story_script, wonder_script, action_script, wisdom_script, etc.)
 *   2. kelly_lesson_assets.script_text for all rows of that day
 * 
 * PUT /api/admin/regenerate-scripts
 * Body: { startDay: 37, endDay: 45 }
 * Batch mode: regenerates a range of days sequentially
 * 
 * GET /api/admin/regenerate-scripts?day=37
 * Status check: shows current scripts vs title for a day
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { generateText, Output } from 'ai'
import { z } from 'zod'

// Master curriculum lookup (imported at build time)
import curriculumLookup from '@/data/curriculum_lookup.json'

const CURRICULUM: Record<string, { topic: string; date: string; learning_objective: string }> = curriculumLookup

// The schema for the AI-generated scripts
const LessonScriptsSchema = z.object({
  hook_script: z.string().describe('A compelling opening question that makes the listener curious about this topic. 1-2 sentences. Start with a question.'),
  hook_fact: z.string().describe('A single fascinating fact related to the topic that serves as the hook. 1 sentence.'),
  hook_correct_answer: z.boolean().describe('Whether the hook_fact statement is true (true) or false/misleading (false) - used for the true/false voting game.'),
  story_script: z.string().describe('A brief personal-feeling narrative about discovering or experiencing this topic. 1-2 sentences. Use first person.'),
  wonder_script: z.string().describe('The core educational content - the amazing explanation or science behind the topic. 2-3 sentences. Make it vivid and clear.'),
  action_script: z.string().describe('A simple, practical activity the listener can do to explore this topic themselves. 1-2 sentences. Make it doable.'),
  wisdom_script: z.string().describe('A universal life lesson or reflection that connects this topic to the bigger picture of life. 1-2 sentences. Make it memorable.'),
  subtitle: z.string().describe('A one-line subtitle for the lesson (not the title). Something evocative.'),
  quote: z.string().describe('A real, attributed quote from a notable person that relates to this topic.'),
  quote_author: z.string().describe('The author of the quote.'),
})

// Reference examples - what correctly-generated scripts look like
const REFERENCE_EXAMPLES = `
EXAMPLE 1 - Day 1: "The Sun - Our Magnificent Life-Giving Star"
  hook_script: "Have you ever looked up and wondered why the sky is blue?"
  hook_fact: "Blue light scatters more than red light in Earth's atmosphere."
  hook_correct_answer: true
  story_script: "Today I looked up at the sky and asked my first big question!"
  wonder_script: "Sunlight contains all colors mixed together. When it hits tiny molecules in our air, blue light bounces around more than other colors because it travels in shorter, smaller waves!"
  action_script: "Go outside and look at the sky at different times of day."
  wisdom_script: "The biggest questions often have the most beautiful answers."

EXAMPLE 2 - Day 5: "Water - The Amazing Molecule That Makes Life Possible"
  hook_script: "Have you ever wondered where rain comes from?"
  hook_fact: "Rain is recycled ocean water that evaporated into clouds."
  hook_correct_answer: true
  story_script: "I thought clouds were like giant sponges that got squeezed."
  wonder_script: "Water evaporates from oceans and lakes, rises up, forms clouds, and falls back down as rain!"
  action_script: "Put a glass of water in the sun and see how much evaporates in a day."
  wisdom_script: "Water travels in an endless journey, connecting everything on Earth."

EXAMPLE 3 - Day 10: "The Scientific Method - Thinking Like a Detective"
  hook_script: "How do scientists know what's really true?"
  hook_fact: "The scientific method requires experiments that other scientists can repeat."
  hook_correct_answer: true
  story_script: "I once had a theory about why my garden grew better on one side."
  wonder_script: "Scientists use a step-by-step method: observe, question, hypothesize, test, and conclude. The key is that anyone should be able to repeat your experiment and get the same result!"
  action_script: "Pick something you're curious about and design a simple experiment to test it."
  wisdom_script: "The best answers come from asking the right questions."
`

async function generateScriptsForDay(dayNumber: number): Promise<{
  success: boolean
  day: number
  title: string
  scripts?: z.infer<typeof LessonScriptsSchema>
  error?: string
}> {
  const curriculum = CURRICULUM[String(dayNumber)]
  if (!curriculum) {
    return { success: false, day: dayNumber, title: '', error: `Day ${dayNumber} not found in curriculum_lookup.json` }
  }

  const { topic, learning_objective } = curriculum

  const prompt = `You are the script writer for "The Daily Lesson" — a 365-day educational experience hosted by Kelly, a warm, curious AI teacher who makes complex topics accessible to all ages.

TOPIC: ${topic}
DAY: ${dayNumber} of 365
LEARNING OBJECTIVE: ${learning_objective}

Write the 5 phase scripts for this lesson. Each phase serves a specific purpose in the Lesson of the Day format:

1. HOOK: An opening question or surprising fact that grabs attention. Should make the listener go "Wait, really?" or "Huh, I never thought about that!" Keep it to 1-2 sentences.

2. STORY: A brief first-person narrative (as Kelly) about discovering or encountering this topic. Should feel personal and relatable. 1-2 sentences.

3. WONDER: The core educational content. This is where the real learning happens — explain the science, the history, or the mechanism in a way that creates genuine awe. 2-3 sentences. Be vivid and specific.

4. ACTION: A simple, practical activity the listener can actually do to explore this topic. Must be doable without special equipment. 1-2 sentences.

5. WISDOM: A universal reflection or life lesson that connects this specific topic to the bigger picture of human experience. Should feel quotable and memorable. 1-2 sentences.

Also provide:
- A hook_fact (1 fascinating true or false statement for the voting game)
- hook_correct_answer (true/false - whether the fact statement is accurate)
- A subtitle (one evocative line)
- A relevant quote from a real notable person
- The quote author

STYLE GUIDE:
- Conversational, warm, wonder-filled tone
- Accessible to ages 6-102 (the "default" voice before age/archetype adaptation)
- Short, punchy sentences. No academic language.
- Scientifically accurate but emotionally engaging.
- Each script should work as standalone spoken audio (Kelly speaks these on camera)

Here are reference examples of the correct style and length:
${REFERENCE_EXAMPLES}

Generate scripts for: "${topic}"`

  try {
    const result = await generateText({
      model: 'openai/gpt-4o-mini',
      prompt,
      output: Output.object({ schema: LessonScriptsSchema }),
    })

    const scripts = result.output
    if (!scripts) {
      return { success: false, day: dayNumber, title: topic, error: 'AI returned no structured output' }
    }

    return { success: true, day: dayNumber, title: topic, scripts }
  } catch (error) {
    return { success: false, day: dayNumber, title: topic, error: String(error) }
  }
}

async function saveScriptsToDb(dayNumber: number, scripts: z.infer<typeof LessonScriptsSchema>) {
  const sql = getSql()

  // 1. Update the lessons table
  await sql`
    UPDATE lessons SET
      hook_script = ${scripts.hook_script},
      story_script = ${scripts.story_script},
      wonder_script = ${scripts.wonder_script},
      action_script = ${scripts.action_script},
      wisdom_script = ${scripts.wisdom_script},
      hook_fact = ${scripts.hook_fact},
      hook_correct_answer = ${scripts.hook_correct_answer},
      subtitle = ${scripts.subtitle},
      quote = ${scripts.quote},
      quote_author = ${scripts.quote_author},
      content_version = COALESCE(content_version, 0) + 1,
      updated_at = NOW()
    WHERE day_of_year = ${dayNumber}
  `

  // 2. Update kelly_lesson_assets script_text for each phase
  const phaseScripts: Record<string, string> = {
    hook: scripts.hook_script,
    story: scripts.story_script,
    wonder: scripts.wonder_script,
    action: scripts.action_script,
    wisdom: scripts.wisdom_script,
  }

  for (const [phase, scriptText] of Object.entries(phaseScripts)) {
    await sql`
      UPDATE kelly_lesson_assets 
      SET script_text = ${scriptText},
          audio_url = NULL,
          video_url = NULL,
          video_id = NULL,
          fal_request_id = NULL,
          status = 'pending',
          updated_at = NOW()
      WHERE day_number = ${dayNumber} AND phase = ${phase}
    `
  }
}

// POST: Generate scripts for a single day
export async function POST(request: NextRequest) {
  try {
    const { day } = await request.json()

    if (!day || day < 1 || day > 365) {
      return NextResponse.json({ error: 'Invalid day (1-365)' }, { status: 400 })
    }

    const result = await generateScriptsForDay(day)

    if (!result.success || !result.scripts) {
      return NextResponse.json({
        error: result.error,
        day,
        title: result.title,
      }, { status: 500 })
    }

    // Save to database
    await saveScriptsToDb(day, result.scripts)

    return NextResponse.json({
      success: true,
      day,
      title: result.title,
      scripts: result.scripts,
      message: `Day ${day} scripts regenerated and saved. kelly_lesson_assets audio/video cleared for re-generation.`,
    })
  } catch (error) {
    console.error('[admin/regenerate-scripts] POST error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// PUT: Batch regenerate a range of days
export async function PUT(request: NextRequest) {
  try {
    const { startDay, endDay } = await request.json()

    if (!startDay || !endDay || startDay < 1 || endDay > 365 || startDay > endDay) {
      return NextResponse.json({ error: 'Invalid range. Use startDay/endDay (1-365).' }, { status: 400 })
    }

    const results: Array<{ day: number; title: string; success: boolean; error?: string }> = []

    for (let day = startDay; day <= endDay; day++) {
      const result = await generateScriptsForDay(day)

      if (result.success && result.scripts) {
        await saveScriptsToDb(day, result.scripts)
        results.push({ day, title: result.title, success: true })
      } else {
        results.push({ day, title: result.title, success: false, error: result.error })
      }

      // Rate limit: 500ms between days
      if (day < endDay) {
        await new Promise(r => setTimeout(r, 500))
      }
    }

    const succeeded = results.filter(r => r.success).length
    const failed = results.filter(r => !r.success).length

    return NextResponse.json({
      success: failed === 0,
      range: `${startDay}-${endDay}`,
      total: results.length,
      succeeded,
      failed,
      results,
    })
  } catch (error) {
    console.error('[admin/regenerate-scripts] PUT error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// GET: Status check for a day
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const dayParam = searchParams.get('day')

  if (!dayParam) {
    return NextResponse.json({ error: 'Provide ?day=N' }, { status: 400 })
  }

  const day = parseInt(dayParam)
  const sql = getSql()

  const lesson = await sql`
    SELECT day_of_year, title, topic, theme, 
           hook_script, story_script, wonder_script, action_script, wisdom_script,
           hook_fact, hook_correct_answer, subtitle, quote, quote_author,
           content_version, updated_at
    FROM lessons WHERE day_of_year = ${day}
  `

  const curriculum = CURRICULUM[String(day)]

  const assetStats = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(audio_url) as with_audio,
      COUNT(video_url) as with_video,
      COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending
    FROM kelly_lesson_assets WHERE day_number = ${day}
  `

  // Check if scripts match the topic (simple heuristic: does hook_script mention the title's key word?)
  const lessonData = lesson[0] as Record<string, unknown> | undefined
  const topicWords = (lessonData?.topic as string || '').toLowerCase().split(/\s+/).filter(w => w.length > 4)
  const hookScript = (lessonData?.hook_script as string || '').toLowerCase()
  const scriptsAligned = topicWords.some(word => hookScript.includes(word)) || !lessonData?.hook_script

  return NextResponse.json({
    day,
    title: lessonData?.title,
    topic: lessonData?.topic,
    curriculum_topic: curriculum?.topic,
    curriculum_objective: curriculum?.learning_objective,
    scripts_aligned: scriptsAligned,
    current_scripts: {
      hook_script: lessonData?.hook_script,
      story_script: lessonData?.story_script,
      wonder_script: lessonData?.wonder_script,
      action_script: lessonData?.action_script,
      wisdom_script: lessonData?.wisdom_script,
    },
    metadata: {
      hook_fact: lessonData?.hook_fact,
      subtitle: lessonData?.subtitle,
      quote: lessonData?.quote,
      quote_author: lessonData?.quote_author,
      content_version: lessonData?.content_version,
      updated_at: lessonData?.updated_at,
    },
    kelly_assets: assetStats[0],
  })
}
