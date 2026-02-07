import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { createClient } from '@supabase/supabase-js'

/**
 * CRITICAL: Lessons Sync API
 * 
 * Syncs REAL curriculum from Supabase (source of truth) to Neon (app database)
 * 
 * TWO TRACKS:
 * - LEARN: 365 daily lessons (20-year curated curriculum) - PRIORITY
 * - GROW: AI track (to be built later)
 * 
 * The Neon database was contaminated with placeholder content (popcorn/dolphins).
 * This API syncs the REAL curriculum from Supabase to fix it.
 */

interface SupabaseLesson {
  id: number
  day_of_year?: number
  day?: number
  title: string
  description?: string
  theme?: string
  subject?: string
  category?: string
  topic_id?: string
  lesson_id?: string
  created_at?: string
  updated_at?: string
}

export async function POST(request: NextRequest) {
  const sql = getSql()
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY
  
  if (!supabaseUrl || !supabaseKey) {
    return NextResponse.json({ error: 'Supabase not configured' }, { status: 500 })
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey)
  
  const { searchParams } = new URL(request.url)
  const specificDay = searchParams.get('day')
  const dryRun = searchParams.get('dry_run') === 'true'

  // Verify authorization
  const authHeader = request.headers.get('authorization')
  const cronSecret = process.env.CRON_SECRET
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    // First, discover what lessons table exists in Supabase
    // Try multiple possible table names
    const possibleTables = [
      'lessons', 'curriculum', 'daily_lessons', 'topics', 'lesson_topics',
      'learn_lessons', 'curriculum_lessons', 'lesson_plans', 'daily_topics',
      'learning_topics', 'education_lessons', 'content', 'lesson_content'
    ]
    let lessonsData: SupabaseLesson[] = []
    let sourceTable = ''
    
    for (const tableName of possibleTables) {
      try {
        let query = supabase
          .from(tableName)
          .select('*')
          .order('id', { ascending: true })
        
        if (specificDay) {
          query = query.or(`day_of_year.eq.${specificDay},day.eq.${specificDay}`)
        }
        
        const { data, error } = await query.limit(400)
        
        if (!error && data && data.length > 0) {
          lessonsData = data as SupabaseLesson[]
          sourceTable = tableName
          break
        }
      } catch {
        // Table doesn't exist, try next
        continue
      }
    }
    
    if (lessonsData.length === 0) {
      return NextResponse.json({ 
        error: 'No lessons found in Supabase',
        checkedTables: possibleTables,
        hint: 'Check Supabase schema for correct table name'
      }, { status: 404 })
    }

    // If dry run, just return what we found
    if (dryRun) {
      return NextResponse.json({
        success: true,
        dryRun: true,
        sourceTable,
        lessonsFound: lessonsData.length,
        sample: lessonsData.slice(0, 5).map(l => ({
          day: l.day_of_year || l.day,
          title: l.title,
          theme: l.theme || l.subject || l.category
        }))
      })
    }

    let synced = 0
    let updated = 0
    const errors: string[] = []

    for (const lesson of lessonsData) {
      try {
        const dayNumber = lesson.day_of_year || lesson.day
        if (!dayNumber) {
          errors.push(`Lesson ${lesson.id} has no day number`)
          continue
        }

        // Check if lesson exists in Neon
        const existing = await sql`
          SELECT id, title FROM lessons WHERE day_of_year = ${dayNumber}
        `

        if (existing.length > 0) {
          // Update existing lesson with REAL data
          await sql`
            UPDATE lessons SET
              title = ${lesson.title},
              description = ${lesson.description || null},
              theme = ${lesson.theme || lesson.subject || lesson.category || null},
              lesson_id = ${lesson.lesson_id || lesson.topic_id || null},
              verified = true,
              curriculum_source = 'supabase_sync',
              track = 'learn',
              updated_at = NOW()
            WHERE day_of_year = ${dayNumber}
          `
          updated++
        } else {
          // Insert new lesson
          await sql`
            INSERT INTO lessons (
              day_of_year, title, description, theme, lesson_id, 
              verified, curriculum_source, track, created_at
            ) VALUES (
              ${dayNumber}, ${lesson.title}, ${lesson.description || null},
              ${lesson.theme || lesson.subject || lesson.category || null},
              ${lesson.lesson_id || lesson.topic_id || null},
              true, 'supabase_sync', 'learn', NOW()
            )
          `
          synced++
        }
      } catch (err) {
        const dayNum = lesson.day_of_year || lesson.day
        errors.push(`Day ${dayNum}: ${err instanceof Error ? err.message : 'Unknown'}`)
      }
    }

    return NextResponse.json({
      success: true,
      sourceTable,
      totalFromSupabase: lessonsData.length,
      synced,
      updated,
      errors: errors.slice(0, 20)
    })
  } catch (error) {
    return NextResponse.json({ 
      error: 'Sync failed', 
      details: error instanceof Error ? error.message : 'Unknown' 
    }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  const sql = getSql()
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY
  
  const { searchParams } = new URL(request.url)
  const checkSupabase = searchParams.get('check_supabase') === 'true'

  try {
    // Get Neon status
    const neonStats = await sql`
      SELECT 
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE verified = true) as verified,
        COUNT(*) FILTER (WHERE curriculum_source = 'supabase_sync') as from_supabase,
        COUNT(*) FILTER (WHERE curriculum_source = 'api_export_contaminated') as contaminated
      FROM lessons
    `
    
    const response: Record<string, unknown> = {
      neon: {
        total: neonStats[0]?.total || 0,
        verified: neonStats[0]?.verified || 0,
        fromSupabase: neonStats[0]?.from_supabase || 0,
        contaminated: neonStats[0]?.contaminated || 0
      }
    }

    // Optionally check Supabase
    if (checkSupabase && supabaseUrl && supabaseKey) {
      const supabase = createClient(supabaseUrl, supabaseKey)
      
      // Try to find lessons table - check all possible names
      const possibleTables = [
        'lessons', 'curriculum', 'daily_lessons', 'topics', 'lesson_topics',
        'learn_lessons', 'curriculum_lessons', 'lesson_plans', 'daily_topics',
        'learning_topics', 'education_lessons', 'content', 'lesson_content'
      ]
      for (const tableName of possibleTables) {
        try {
          const { count, error } = await supabase
            .from(tableName)
            .select('*', { count: 'exact', head: true })
          
          if (!error && count !== null) {
            response.supabase = {
              table: tableName,
              count
            }
            break
          }
        } catch {
          continue
        }
      }
    }

    return NextResponse.json(response)
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown' 
    }, { status: 500 })
  }
}
