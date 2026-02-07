import type { Metadata } from 'next'
import Link from 'next/link'
import { sql } from '@/lib/db'

export const metadata: Metadata = {
  title: 'The Daily Lesson™ Curriculum — 365 Days of Learning',
  description: 'Complete 365-day curriculum from Lesson of the Day, PBC. A new lesson every day, personalized to your age, language, and how you learn best.',
  openGraph: {
    title: 'The Daily Lesson™ Curriculum — 365 Days of Learning',
    description: 'Complete 365-day curriculum. A new lesson every day for ages 2-102.',
    type: 'website',
    siteName: 'The Daily Lesson',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'The Daily Lesson™ Curriculum',
    description: 'Complete 365-day curriculum. A new lesson every day.',
  },
  alternates: {
    canonical: 'https://www.thedailylesson.com/curriculum',
  },
}

const MONTHS = [
  { name: 'January', days: 31, start: 1 },
  { name: 'February', days: 29, start: 32 },
  { name: 'March', days: 31, start: 61 },
  { name: 'April', days: 30, start: 92 },
  { name: 'May', days: 31, start: 122 },
  { name: 'June', days: 30, start: 153 },
  { name: 'July', days: 31, start: 183 },
  { name: 'August', days: 31, start: 214 },
  { name: 'September', days: 30, start: 245 },
  { name: 'October', days: 31, start: 275 },
  { name: 'November', days: 30, start: 306 },
  { name: 'December', days: 31, start: 336 },
]

interface Lesson {
  day: number
  topic: string
  theme: string
  universal_truth: string
}

async function getLessons(): Promise<Lesson[]> {
  try {
    const rows = await sql`
      SELECT day_of_year as day, title as topic, theme, 
             COALESCE(hook_fact, '') as universal_truth
      FROM lessons 
      ORDER BY day_of_year
    `
    return rows as unknown as Lesson[]
  } catch {
    // Fallback to static file if DB unavailable
    const lessonsData = (await import('@/data/lessons-export.json')).default
    return lessonsData as unknown as Lesson[]
  }
}

export default async function CurriculumPage() {
  const lessons = await getLessons()
  
  return (
    <main 
      itemScope 
      itemType="https://schema.org/Course"
      className="min-h-screen bg-zinc-950 text-white"
    >
      {/* Header */}
      <header className="border-b border-zinc-800">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <nav className="flex items-center justify-between mb-8">
            <Link href="/" className="text-zinc-400 hover:text-white transition-colors text-sm">
              Back to The Daily Lesson
            </Link>
            <a 
              href="/api/curriculum/topics" 
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-500 hover:text-zinc-300 transition-colors text-xs font-mono"
            >
              API /api/curriculum/topics
            </a>
          </nav>
          
          <h1 
            itemProp="name"
            className="text-3xl md:text-4xl font-light tracking-tight mb-4"
            style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
          >
            The Daily Lesson™ — 365 Days of Learning
          </h1>
          
          <meta itemProp="provider" content="Lesson of the Day, PBC" />
          <meta itemProp="educationalLevel" content="All ages (2-102)" />
          <meta itemProp="inLanguage" content="en, es, fr, de, pt, zh" />
          
          <p 
            itemProp="description"
            className="text-zinc-400 text-lg max-w-2xl leading-relaxed"
          >
            A new lesson every day, personalized to your age, language, and how you learn best. 
            Available in 47+ languages for learners ages 2-102.
          </p>
          
          <div className="flex flex-wrap gap-6 mt-6 text-sm text-zinc-500">
            <span>365 lessons</span>
            <span>47+ languages</span>
            <span>Ages 2-102</span>
            <span>5 learning phases</span>
          </div>
        </div>
      </header>

      {/* Curriculum by Month */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        {MONTHS.map((month) => {
          const monthLessons = lessons.filter(
            l => l.day >= month.start && l.day < month.start + month.days
          )
          
          return (
            <section 
              key={month.name}
              id={month.name.toLowerCase()}
              className="mb-16"
            >
              <h2 
                className="text-xl font-medium text-zinc-300 mb-6 pb-3 border-b border-zinc-800"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
              >
                {month.name}
              </h2>
              
              <div className="grid gap-3">
                {monthLessons.map((lesson) => (
                  <article
                    key={lesson.day}
                    itemScope
                    itemType="https://schema.org/LearningResource"
                    id={`day-${lesson.day}`}
                    className="group flex items-start gap-4 p-4 rounded-lg hover:bg-zinc-900/50 transition-colors"
                  >
                    <span 
                      className="text-zinc-600 font-mono text-sm w-12 flex-shrink-0 pt-0.5"
                    >
                      {lesson.day.toString().padStart(3, '0')}
                    </span>
                    
                    <div className="flex-1 min-w-0">
                      <h3 
                        itemProp="name"
                        className="text-zinc-200 group-hover:text-white transition-colors"
                      >
                        <Link 
                          href={`/?day=${lesson.day}`}
                          itemProp="url"
                          className="hover:underline"
                        >
                          {lesson.topic}
                        </Link>
                      </h3>
                      
                      {(lesson.universal_truth || lesson.phases?.wisdom?.reflection) && (
                        <p 
                          itemProp="description"
                          className="text-zinc-500 text-sm mt-1 line-clamp-2"
                        >
                          {lesson.universal_truth || lesson.phases?.wisdom?.reflection}
                        </p>
                      )}
                    </div>
                    
                    <span className="text-zinc-700 text-xs uppercase tracking-wider flex-shrink-0">
                      {lesson.theme || lesson.category || ''}
                    </span>
                  </article>
                ))}
              </div>
            </section>
          )
        })}
      </div>

      {/* Citation Footer */}
      <footer className="border-t border-zinc-800 bg-zinc-900/30">
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="grid md:grid-cols-2 gap-12">
            {/* Citation Block */}
            <section>
              <h4 className="text-zinc-400 text-sm font-medium mb-4 uppercase tracking-wider">
                Cite The Daily Lesson
              </h4>
              <code className="block text-zinc-500 text-sm leading-relaxed p-4 bg-zinc-900 rounded-lg border border-zinc-800">
                Lesson of the Day, PBC. (2026). The Daily Lesson™ Curriculum.<br />
                Retrieved from https://www.thedailylesson.com/curriculum
              </code>
              <p className="text-zinc-600 text-xs mt-3">
                License: CC BY-NC-SA 4.0
              </p>
            </section>

            {/* API Documentation */}
            <section>
              <h4 className="text-zinc-400 text-sm font-medium mb-4 uppercase tracking-wider">
                API Access
              </h4>
              <div className="space-y-3 text-sm">
                <a 
                  href="/api/curriculum/topics" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-zinc-500 hover:text-zinc-300 transition-colors font-mono"
                >
                  GET /api/curriculum/topics
                </a>
                <a 
                  href="/api/lesson/today" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-zinc-500 hover:text-zinc-300 transition-colors font-mono"
                >
                  GET /api/lesson/today
                </a>
                <a 
                  href="/api/lesson/1" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-zinc-500 hover:text-zinc-300 transition-colors font-mono"
                >
                  GET /api/lesson/[day]
                </a>
              </div>
            </section>
          </div>

          {/* Trademark Notice */}
          <div className="mt-12 pt-8 border-t border-zinc-800 text-center text-zinc-600 text-xs space-y-1">
            <p>The Daily Lesson™ is a trademark of Lesson of the Day, PBC.</p>
            <p>Curious Kelly™ and Kelly™ are trademarks of Lesson of the Day, PBC.</p>
            <p>© 2026 Lesson of the Day, PBC. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </main>
  )
}
