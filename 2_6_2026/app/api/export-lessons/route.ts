import { getSql } from '@/lib/db'

export async function GET() {
  const sql = getSql()
  const lessons = await sql`
    SELECT 
      day_of_year as day,
      title as topic,
      theme,
      kelly_age,
      hook_script,
      story_script,
      wonder_script,
      action_script,
      quote,
      quote_author,
      wisdom_script
    FROM lessons 
    ORDER BY day_of_year
  `

  // Transform to nested phases structure for visual prompt factory
  const formatted = lessons.map((lesson: any) => ({
    day: lesson.day,
    topic: lesson.topic,
    theme: lesson.theme,
    kelly_age: lesson.kelly_age,
    phases: {
      hook: { question: lesson.hook_script },
      story: { narrative: lesson.story_script },
      wonder: { exploration: lesson.wonder_script },
      action: { activity: lesson.action_script },
      wisdom: { 
        quote: lesson.quote, 
        quote_author: lesson.quote_author,
        reflection: lesson.wisdom_script 
      }
    }
  }))

  return new Response(JSON.stringify(formatted, null, 2), {
    headers: {
      'Content-Type': 'application/json',
      'Content-Disposition': 'attachment; filename="lessons-export.json"'
    }
  })
}
