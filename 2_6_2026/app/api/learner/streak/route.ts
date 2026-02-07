import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { cookies } from 'next/headers'

export const dynamic = 'force-dynamic'

interface StreakSyncRequest {
  localStreak?: number
  localCompletedDays?: number[]
}

// GET - Retrieve current streak data
export async function GET() {
  try {
    const cookieStore = await cookies()
    const fingerprint = cookieStore.get('ck_fp')?.value

    if (!fingerprint) {
      return NextResponse.json({
        currentStreak: 0,
        longestStreak: 0,
        lastLessonDate: null,
        completedDaysThisYear: []
      })
    }

    const currentYear = new Date().getFullYear()

    // Get learner data
    const learnerData = await sql`
      SELECT current_streak, longest_streak, last_lesson_date
      FROM learners WHERE fingerprint = ${fingerprint}
    `

    // Get completed days this year (days where all 5 phases are done)
    const completedDaysResult = await sql`
      SELECT day_of_year, COUNT(DISTINCT phase) as phase_count
      FROM learner_progress 
      WHERE fingerprint = ${fingerprint} AND year = ${currentYear}
      GROUP BY day_of_year
      HAVING COUNT(DISTINCT phase) = 5
      ORDER BY day_of_year
    `

    const completedDaysThisYear = completedDaysResult.map(r => r.day_of_year as number)

    if (learnerData.length > 0) {
      const { current_streak, longest_streak, last_lesson_date } = learnerData[0]
      return NextResponse.json({
        currentStreak: Number(current_streak) || 0,
        longestStreak: Number(longest_streak) || 0,
        lastLessonDate: last_lesson_date,
        completedDaysThisYear
      })
    }

    return NextResponse.json({
      currentStreak: 0,
      longestStreak: 0,
      lastLessonDate: null,
      completedDaysThisYear
    })

  } catch (error) {
    console.error('[streak] GET Error:', error)
    return NextResponse.json({
      currentStreak: 0,
      longestStreak: 0,
      lastLessonDate: null,
      completedDaysThisYear: []
    })
  }
}

// POST - Sync local streak data to server (migration from localStorage)
export async function POST(request: NextRequest) {
  try {
    const cookieStore = await cookies()
    const fingerprint = cookieStore.get('ck_fp')?.value

    if (!fingerprint) {
      return NextResponse.json({ error: 'No fingerprint found' }, { status: 401 })
    }

    const body: StreakSyncRequest = await request.json()
    const { localStreak = 0, localCompletedDays = [] } = body

    // Get or create learner
    const learnerResult = await sql`
      INSERT INTO learners (fingerprint, created_at)
      VALUES (${fingerprint}, now())
      ON CONFLICT (fingerprint) DO UPDATE SET fingerprint = EXCLUDED.fingerprint
      RETURNING id, current_streak, longest_streak
    `
    
    const learner = learnerResult[0]
    const learnerId = learner?.id as string
    const serverStreak = Number(learner?.current_streak) || 0
    const serverLongest = Number(learner?.longest_streak) || 0

    // Only migrate if local data is richer than server data
    const shouldMigrate = localStreak > serverStreak || localCompletedDays.length > 0

    if (shouldMigrate && learnerId) {
      const currentYear = new Date().getFullYear()
      
      // Migrate completed days to learner_progress
      // We'll mark all 5 phases as complete for migrated days
      for (const dayOfYear of localCompletedDays) {
        if (dayOfYear >= 1 && dayOfYear <= 366) {
          for (const phase of ['hook', 'story', 'wonder', 'action', 'wisdom']) {
            try {
              await sql`
                INSERT INTO learner_progress (learner_id, fingerprint, day_of_year, phase, year)
                VALUES (${learnerId}, ${fingerprint}, ${dayOfYear}, ${phase}, ${currentYear})
                ON CONFLICT (fingerprint, day_of_year, phase, year) DO NOTHING
              `
            } catch (e) {
              // Ignore individual insert failures
            }
          }
        }
      }

      // Update streak if local is higher
      const newStreak = Math.max(localStreak, serverStreak)
      const newLongest = Math.max(localStreak, serverLongest)
      
      await sql`
        UPDATE learners 
        SET current_streak = ${newStreak},
            longest_streak = ${newLongest},
            streak_updated_at = now()
        WHERE id = ${learnerId}
      `

      return NextResponse.json({
        migrated: true,
        currentStreak: newStreak,
        longestStreak: newLongest,
        migratedDays: localCompletedDays.length
      })
    }

    return NextResponse.json({
      migrated: false,
      currentStreak: serverStreak,
      longestStreak: serverLongest,
      migratedDays: 0
    })

  } catch (error) {
    console.error('[streak] POST Error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
