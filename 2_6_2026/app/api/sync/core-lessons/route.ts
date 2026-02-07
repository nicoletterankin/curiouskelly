import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { createClient } from '@supabase/supabase-js'

/**
 * Sync 730 lessons from Supabase core_lessons to Neon
 * 
 * POST /api/sync/core-lessons - Sync all lessons
 * GET /api/sync/core-lessons - Check sync status
 */

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co'
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI'

interface CoreLesson {
  id: string
  day_number: number
  track: 'learn' | 'grow'
  topic: string
  universal_truth: string
  icon_emoji: string
  marketing_headline: string
  marketing_tagline: string
  marketing_pitch: string
  learning_objectives: unknown[]
  fun_facts: unknown[]
  quick_quiz_questions: unknown[]
  reflection_prompts: unknown[]
}

export async function GET() {
  try {
    const sql = getSql()
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    // Check Neon status
    const neonStats = await sql`
      SELECT 
        COUNT(*)::int as total,
        COUNT(DISTINCT day_of_year)::int as unique_days,
        COUNT(*) FILTER (WHERE track = 'learn')::int as learn_count,
        COUNT(*) FILTER (WHERE track = 'grow')::int as grow_count,
        COUNT(*) FILTER (WHERE curriculum_source = 'supabase_core_lessons')::int as from_supabase,
        MIN(day_of_year)::int as min_day,
        MAX(day_of_year)::int as max_day
      FROM lessons
    `
    
    // Check Supabase status
    const { count: supabaseCount } = await supabase
      .from('core_lessons')
      .select('*', { count: 'exact', head: true })
    
    // Find missing days
    const existingDays = await sql`
      SELECT DISTINCT day_of_year FROM lessons WHERE track = 'learn' ORDER BY day_of_year
    `
    const existingDaySet = new Set(existingDays.map(d => d.day_of_year))
    const missingDays: number[] = []
    for (let day = 1; day <= 365; day++) {
      if (!existingDaySet.has(day)) {
        missingDays.push(day)
      }
    }
    
    return NextResponse.json({
      neon: {
        total: neonStats[0]?.total || 0,
        uniqueDays: neonStats[0]?.unique_days || 0,
        learnCount: neonStats[0]?.learn_count || 0,
        growCount: neonStats[0]?.grow_count || 0,
        fromSupabase: neonStats[0]?.from_supabase || 0,
        dayRange: `${neonStats[0]?.min_day || 0}-${neonStats[0]?.max_day || 0}`,
      },
      supabase: {
        count: supabaseCount || 0,
        expected: 730,
        complete: supabaseCount === 730
      },
      missingDays: {
        count: missingDays.length,
        days: missingDays.slice(0, 20),
        hasMore: missingDays.length > 20
      },
      needsSync: missingDays.length > 0
    })
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const sql = getSql()
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    const { searchParams } = new URL(request.url)
    const dryRun = searchParams.get('dry_run') === 'true'
    const trackFilter = searchParams.get('track') as 'learn' | 'grow' | null
    const dayFilter = searchParams.get('day')
    
    // Build Supabase query
    let query = supabase
      .from('core_lessons')
      .select('*')
      .order('day_number')
      .order('track')
    
    if (trackFilter) {
      query = query.eq('track', trackFilter)
    }
    if (dayFilter) {
      query = query.eq('day_number', parseInt(dayFilter))
    }
    
    const { data: lessons, error } = await query
    
    if (error) {
      return NextResponse.json({ 
        error: `Supabase fetch failed: ${error.message}` 
      }, { status: 500 })
    }
    
    if (!lessons || lessons.length === 0) {
      return NextResponse.json({ 
        error: 'No lessons found in Supabase core_lessons table' 
      }, { status: 404 })
    }
    
    if (dryRun) {
      return NextResponse.json({
        dryRun: true,
        lessonsFound: lessons.length,
        sample: lessons.slice(0, 5).map((l: CoreLesson) => ({
          day: l.day_number,
          track: l.track,
          topic: l.topic,
          icon: l.icon_emoji
        }))
      })
    }
    
    // Ensure lessons table has required columns
    await sql`
      ALTER TABLE lessons 
      ADD COLUMN IF NOT EXISTS topic TEXT,
      ADD COLUMN IF NOT EXISTS icon TEXT,
      ADD COLUMN IF NOT EXISTS learning_objectives JSONB,
      ADD COLUMN IF NOT EXISTS fun_facts JSONB,
      ADD COLUMN IF NOT EXISTS quiz_questions JSONB,
      ADD COLUMN IF NOT EXISTS reflection_prompts JSONB,
      ADD COLUMN IF NOT EXISTS marketing_headline TEXT,
      ADD COLUMN IF NOT EXISTS marketing_tagline TEXT
    `.catch(() => {
      // Columns might already exist or table structure differs
    })
    
    let inserted = 0
    let updated = 0
    const errors: string[] = []
    
    for (const lesson of lessons as CoreLesson[]) {
      try {
        // Check if exists
        const existing = await sql`
          SELECT id FROM lessons 
          WHERE day_of_year = ${lesson.day_number} 
          AND track = ${lesson.track}
        `
        
        if (existing.length > 0) {
          // Update
          await sql`
            UPDATE lessons SET
              topic = ${lesson.topic},
              theme = ${lesson.universal_truth},
              title = ${lesson.topic},
              description = ${lesson.marketing_pitch || lesson.marketing_headline || ''},
              icon = ${lesson.icon_emoji},
              verified = true,
              curriculum_source = 'supabase_core_lessons',
              updated_at = NOW()
            WHERE day_of_year = ${lesson.day_number} AND track = ${lesson.track}
          `
          updated++
        } else {
          // Insert
          await sql`
            INSERT INTO lessons (
              day_of_year, track, topic, theme, title, description, icon,
              verified, curriculum_source, created_at
            ) VALUES (
              ${lesson.day_number}, ${lesson.track}, ${lesson.topic},
              ${lesson.universal_truth}, ${lesson.topic},
              ${lesson.marketing_pitch || lesson.marketing_headline || ''},
              ${lesson.icon_emoji},
              true, 'supabase_core_lessons', NOW()
            )
          `
          inserted++
        }
      } catch (err) {
        errors.push(`Day ${lesson.day_number} (${lesson.track}): ${err instanceof Error ? err.message : 'Unknown'}`)
      }
    }
    
    return NextResponse.json({
      success: true,
      source: 'supabase:core_lessons',
      lessonsProcessed: lessons.length,
      inserted,
      updated,
      errors: errors.length,
      errorSample: errors.slice(0, 10)
    })
    
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 })
  }
}
