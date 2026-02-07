/**
 * ADMIN: Generate Kelly variant scripts (kid/adult/elder) for kelly_scripts table
 *
 * POST /api/admin/kelly-scripts/generate
 * Body: { day: 37 }                           – single day, all 3 variants
 * Body: { day: 37, variant: "kid" }            – single day, single variant
 * Body: { startDay: 37, endDay: 42 }           – batch range, all variants
 *
 * GET /api/admin/kelly-scripts/generate?status=true
 * Returns pipeline progress from kelly_pipeline_progress view
 *
 * Uses the prompt template and spec from the directive:
 *   lessons.topic + lessons.learning_objective → GPT-4o-mini → kelly_scripts table
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { generateText, Output } from 'ai'
import { z } from 'zod'

// Import curriculum lookup for topic data
import curriculumLookup from '@/data/curriculum_lookup.json'

const CURRICULUM: Record<
  string,
  { topic: string; date: string; learning_objective: string }
> = curriculumLookup

// ---------- Variant spec (from KELLY_SCRIPT_GENERATION_SPEC) ----------

const VARIANT_SPEC: Record<
  string,
  {
    age_range: string
    tone: string
    vocabulary: string
    sentence_length: string
    word_counts: Record<string, string>
  }
> = {
  kid: {
    age_range: '2-10',
    tone: 'Warm, playful, wonder-filled. Short sentences. Concrete examples. "Wow!" and "Guess what?" energy. Never condescending.',
    vocabulary:
      'Simple, vivid words. Analogies to playground, family, animals, food.',
    sentence_length: '5-12 words average',
    word_counts: {
      hook: '30-50',
      story: '80-120',
      wonder: '40-60',
      action: '40-60',
      wisdom: '20-40',
    },
  },
  adult: {
    age_range: '18-64',
    tone: 'Conversational, intellectually engaging, respectful. Like a brilliant friend explaining over coffee.',
    vocabulary:
      'Full vocabulary, technical terms introduced naturally with context.',
    sentence_length: '10-20 words average',
    word_counts: {
      hook: '40-70',
      story: '120-180',
      wonder: '60-90',
      action: '50-80',
      wisdom: '30-50',
    },
  },
  elder: {
    age_range: '65-102',
    tone: 'Respectful, reflective, connecting to lived experience. Honors wisdom they already carry. Never patronizing.',
    vocabulary:
      'Rich, measured. References to historical context, legacy, long perspective.',
    sentence_length: '8-18 words average',
    word_counts: {
      hook: '35-60',
      story: '100-150',
      wonder: '50-80',
      action: '45-70',
      wisdom: '30-50',
    },
  },
}

// ---------- Zod schema for AI output ----------

const VariantScriptsSchema = z.object({
  hook: z
    .string()
    .describe(
      "Kelly's opening script — grab attention, create curiosity gap. Spoken directly to learner."
    ),
  story: z
    .string()
    .describe(
      "Kelly's main teaching script — core concept taught through narrative. This IS the lesson."
    ),
  wonder: z
    .string()
    .describe(
      "Kelly's wonder script — expand minds, connect to bigger picture, inspire awe."
    ),
  action: z
    .string()
    .describe(
      "Kelly's action script — specific, doable activity to try today."
    ),
  wisdom: z
    .string()
    .describe(
      "Kelly's closing wisdom — one powerful universal truth as a takeaway."
    ),
})

// ---------- Day 37 reference examples (gold standard) ----------

const REFERENCE_BY_VARIANT: Record<string, string> = {
  kid: `REFERENCE (Day 37, Kid variant):
hook: "Hey! Did you know that music is actually made of math? Yep — when you hear your favorite song, your ears are listening to numbers dancing together! Every note is a vibration, and some vibrations sound amazing when they play at the same time. Want to know why?"
story: "So here's the cool thing — thousands of years ago, a man named Pythagoras discovered that if you take a string and cut it exactly in half, it makes the same note but higher. That's called an octave! It's like magic, but it's actually math. When musicians write songs, they're using patterns and ratios — kind of like building with invisible Legos. A chord is three or more notes that sound good together because their vibrations match up in special ways. Every song you've ever loved? It follows these mathematical patterns."
wonder: "Here's what's amazing — people all over the world, who speak different languages and live totally different lives, ALL think certain note combinations sound beautiful. Math is a language that every human ear already understands!"
action: "Try this! Grab two glasses and fill one halfway with water. Tap them with a spoon. Hear how they make different sounds? You just made music with math! The water changes how fast the glass vibrates."
wisdom: "Music proves that math isn't just numbers on paper — it's the hidden pattern behind everything beautiful."`,

  adult: `REFERENCE (Day 37, Adult variant):
hook: "There's a reason certain chord progressions give you chills — and it has nothing to do with taste or culture. It's mathematics. The relationship between frequencies that we perceive as 'beautiful' follows ratios that Pythagoras mapped twenty-five hundred years ago. Your emotional response to music is, at its core, a response to mathematical harmony."
story: "Every musical note is a frequency — A4 is exactly 440 hertz, meaning the air vibrates 440 times per second. An octave above? Double it: 880 hertz. This doubling relationship is why octaves sound like the 'same note, but higher' to every human ear on Earth. Chords work because their frequency ratios are simple — a perfect fifth is a 3:2 ratio, a major third is 5:4. When these ratios align, the sound waves reinforce each other in patterns your brain finds inherently pleasing. Composers from Bach to Beyoncé are essentially arranging mathematical relationships in time. The Western twelve-tone scale, Indian ragas, Chinese pentatonic scales — all different systems, all built on the same underlying physics of wave interaction."
wonder: "What's remarkable is this: music may be the oldest proof that mathematics isn't a human invention — it's a discovery. The ratios exist whether or not anyone plays a note. We didn't create harmony. We found it."
action: "Pull up a piano app and play C, then E, then G simultaneously — that's a C major chord, built on the ratio 4:5:6. Now play C, E-flat, G — a minor chord, ratio 10:12:15. Feel the emotional difference? That's your brain processing a mathematical shift."
wisdom: "Music is the audible proof that beauty has structure — and that mathematics speaks a language deeper than words."`,

  elder: `REFERENCE (Day 37, Elder variant):
hook: "You've spent a lifetime hearing music — songs that marked your wedding, your losses, your triumphs. But have you ever wondered why certain melodies move every human heart the same way? The answer is older than civilization itself, written into the mathematics of vibration."
story: "When Pythagoras stretched a string and discovered that halving its length produced a note one octave higher, he uncovered something profound: beauty follows proportion. Every note is a frequency, and the combinations we find pleasing — the chords that make us weep or dance — are simple mathematical ratios. A perfect fifth, that satisfying interval in 'Twinkle Twinkle Little Star,' is a 3:2 frequency ratio. This isn't Western or Eastern — it's universal. Cultures across millennia who never met each other independently discovered these same intervals. From Gregorian chants to jazz to the music of the Andes, the mathematics underneath remains constant."
wonder: "Consider this: every lullaby you ever sang to a child was built on ratios that existed before humanity. The mathematics of music preceded us, and will outlast us. We are not creators of harmony — we are its grateful audience."
action: "The next time you hear a piece of music that moves you, pause and listen for the pattern. Notice the intervals, the repetitions, the resolution of tension. You're hearing mathematics expressing itself as emotion — a gift you can appreciate more deeply with each passing year."
wisdom: "A lifetime of listening has taught you what Pythagoras proved — that beauty is not random. It is structured, universal, and eternal."`,
}

// ---------- Generation function ----------

async function generateVariantScripts(
  dayNumber: number,
  variant: string
): Promise<{
  success: boolean
  day: number
  variant: string
  title: string
  scripts?: z.infer<typeof VariantScriptsSchema>
  error?: string
}> {
  const curriculum = CURRICULUM[String(dayNumber)]
  if (!curriculum) {
    return {
      success: false,
      day: dayNumber,
      variant,
      title: '',
      error: `Day ${dayNumber} not found in curriculum_lookup.json`,
    }
  }

  const { topic, learning_objective } = curriculum
  const spec = VARIANT_SPEC[variant]
  if (!spec) {
    return {
      success: false,
      day: dayNumber,
      variant,
      title: topic,
      error: `Unknown variant: ${variant}`,
    }
  }

  const prompt = `You are Kelly, an AI teacher for Curious Kelly (thedailylesson.com). You teach one lesson per day to learners ages 2-102.

TODAY'S LESSON:
- Day: ${dayNumber} of 365
- Title: ${topic}
- Learning Objective: ${learning_objective}

VARIANT: ${variant} (ages ${spec.age_range})

TONE: ${spec.tone}
VOCABULARY: ${spec.vocabulary}
SENTENCE LENGTH: ${spec.sentence_length}

Generate Kelly's word-by-word teaching script for all 5 phases. Kelly speaks directly to the learner in second person. She is warm, knowledgeable, and genuinely passionate about the topic. She NEVER dumbs things down — she makes complex ideas accessible through vivid language and real examples.

Phase requirements:
1. HOOK: Grab attention, create curiosity gap. (${spec.word_counts.hook} words)
2. STORY: Teach the core concept through narrative. This is the LESSON. (${spec.word_counts.story} words)
3. WONDER: Expand minds, connect to bigger picture, inspire awe. (${spec.word_counts.wonder} words)
4. ACTION: Specific activity they can try today, connected to the lesson. (${spec.word_counts.action} words)
5. WISDOM: One powerful universal truth as a takeaway. (${spec.word_counts.wisdom} words)

CRITICAL RULES:
- Kelly NEVER says "I'm an AI" or breaks character
- Kelly ALWAYS speaks in second person with warmth
- Scripts must be scientifically accurate but emotionally engaging
- Each script is spoken by Kelly on camera as audio

${REFERENCE_BY_VARIANT[variant]}

Generate scripts for: "${topic}"`

  try {
    const result = await generateText({
      model: 'openai/gpt-4o-mini',
      prompt,
      output: Output.object({ schema: VariantScriptsSchema }),
    })

    const scripts = result.output
    if (!scripts) {
      return {
        success: false,
        day: dayNumber,
        variant,
        title: topic,
        error: 'AI returned no structured output',
      }
    }

    return { success: true, day: dayNumber, variant, title: topic, scripts }
  } catch (error) {
    return {
      success: false,
      day: dayNumber,
      variant,
      title: topic,
      error: String(error),
    }
  }
}

// ---------- Save to DB ----------

async function saveVariantScripts(
  dayNumber: number,
  variant: string,
  scripts: z.infer<typeof VariantScriptsSchema>,
  model: string = 'gpt-4o-mini'
) {
  const sql = getSql()
  const phases = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const

  for (const phase of phases) {
    const scriptText = scripts[phase]

    // UPSERT: insert or update on conflict
    await sql`
      INSERT INTO kelly_scripts (day_number, variant, phase, script_text, generation_model, generated_at)
      VALUES (${dayNumber}, ${variant}, ${phase}, ${scriptText}, ${model}, NOW())
      ON CONFLICT (day_number, variant, phase)
      DO UPDATE SET
        script_text = EXCLUDED.script_text,
        generation_model = EXCLUDED.generation_model,
        generated_at = NOW(),
        audio_generated = false,
        video_generated = false
    `
  }
}

// ---------- POST: Generate scripts ----------

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { day, variant, startDay, endDay } = body

    // Batch mode
    if (startDay && endDay) {
      if (startDay < 1 || endDay > 365 || startDay > endDay) {
        return NextResponse.json(
          { error: 'Invalid range. Use startDay/endDay (1-365).' },
          { status: 400 }
        )
      }

      const variants = variant ? [variant] : ['kid', 'adult', 'elder']
      const results: Array<{
        day: number
        variant: string
        title: string
        success: boolean
        error?: string
      }> = []

      for (let d = startDay; d <= endDay; d++) {
        for (const v of variants) {
          const result = await generateVariantScripts(d, v)
          if (result.success && result.scripts) {
            await saveVariantScripts(d, v, result.scripts)
            results.push({
              day: d,
              variant: v,
              title: result.title,
              success: true,
            })
          } else {
            results.push({
              day: d,
              variant: v,
              title: result.title,
              success: false,
              error: result.error,
            })
          }
          // Rate limit: 300ms between calls
          await new Promise((r) => setTimeout(r, 300))
        }
      }

      const succeeded = results.filter((r) => r.success).length
      const failed = results.filter((r) => !r.success).length

      return NextResponse.json({
        success: failed === 0,
        mode: 'batch',
        range: `${startDay}-${endDay}`,
        variants,
        total: results.length,
        succeeded,
        failed,
        results,
      })
    }

    // Single day mode
    if (!day || day < 1 || day > 365) {
      return NextResponse.json(
        { error: 'Invalid day (1-365). Or use startDay/endDay for batch.' },
        { status: 400 }
      )
    }

    const variants = variant ? [variant] : ['kid', 'adult', 'elder']
    const results: Array<{
      day: number
      variant: string
      title: string
      success: boolean
      scripts?: z.infer<typeof VariantScriptsSchema>
      error?: string
    }> = []

    for (const v of variants) {
      const result = await generateVariantScripts(day, v)
      if (result.success && result.scripts) {
        await saveVariantScripts(day, v, result.scripts)
        results.push({
          day,
          variant: v,
          title: result.title,
          success: true,
          scripts: result.scripts,
        })
      } else {
        results.push({
          day,
          variant: v,
          title: result.title,
          success: false,
          error: result.error,
        })
      }
      // Rate limit between variants
      if (variants.length > 1) {
        await new Promise((r) => setTimeout(r, 300))
      }
    }

    const allSuccess = results.every((r) => r.success)

    return NextResponse.json({
      success: allSuccess,
      mode: 'single',
      day,
      variants,
      total_scripts: results.length * 5,
      results,
      message: allSuccess
        ? `Day ${day} variant scripts generated and saved to kelly_scripts (${results.length * 5} scripts).`
        : `Some variants failed for day ${day}.`,
    })
  } catch (error) {
    console.error('[admin/kelly-scripts/generate] POST error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

// ---------- GET: Pipeline status ----------

export async function GET() {
  try {
    const sql = getSql()

    // Overall progress
    const progress = await sql`
      SELECT * FROM kelly_pipeline_progress
    `

    // Per-day/variant status (recent 20)
    const recent = await sql`
      SELECT * FROM kelly_scripts_status
      ORDER BY day_number DESC, variant
      LIMIT 60
    `

    // Days with complete scripts (all 3 variants fully scripted)
    const complete = await sql`
      SELECT day_number,
             COUNT(*) FILTER (WHERE fully_scripted) as variants_complete
      FROM kelly_scripts_status
      GROUP BY day_number
      HAVING COUNT(*) FILTER (WHERE fully_scripted) = 3
      ORDER BY day_number
    `

    return NextResponse.json({
      pipeline: progress[0] || {
        days_with_scripts: 0,
        total_scripts: 0,
        audio_done: 0,
        video_done: 0,
        percent_complete: 0,
      },
      fully_complete_days: complete.map(
        (r: Record<string, unknown>) => r.day_number
      ),
      recent_status: recent,
    })
  } catch (error) {
    console.error('[admin/kelly-scripts/generate] GET error:', error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
