import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { cookies } from 'next/headers'

export const dynamic = 'force-dynamic'

const VALID_PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
type Phase = typeof VALID_PHASES[number]

// XP rewards per phase
const PHASE_XP: Record<Phase, number> = {
  hook: 10,
  story: 10,
  wonder: 10,
  action: 15,
  wisdom: 5
}

const DAY_COMPLETION_XP = 50
const WEEKLY_STREAK_BONUS = 100
const MONTHLY_STREAK_BONUS = 500

interface PhaseCompleteRequest {
  dayOfYear: number
  phase: Phase
  timeSpentSeconds?: number
}

export async function POST(request: NextRequest) {
  try {
    const cookieStore = await cookies()
    const fingerprint = cookieStore.get('ck_fp')?.value

    if (!fingerprint) {
      return NextResponse.json({ error: 'No fingerprint found' }, { status: 401 })
    }

    const body: PhaseCompleteRequest = await request.json()
    const { dayOfYear, phase, timeSpentSeconds = 0 } = body

    // Validate inputs
    if (!dayOfYear || dayOfYear < 1 || dayOfYear > 366) {
      return NextResponse.json({ error: 'Invalid dayOfYear' }, { status: 400 })
    }

    if (!phase || !VALID_PHASES.includes(phase)) {
      return NextResponse.json({ error: 'Invalid phase' }, { status: 400 })
    }

    const currentYear = new Date().getFullYear()

    // Get or create learner
    let learnerId: string | null = null
    try {
      const learnerResult = await sql`
        INSERT INTO learners (fingerprint, created_at)
        VALUES (${fingerprint}, now())
        ON CONFLICT (fingerprint) DO UPDATE SET fingerprint = EXCLUDED.fingerprint
        RETURNING id
      `
      learnerId = learnerResult[0]?.id as string
    } catch (e) {
      console.error('[phase-complete] Failed to get/create learner:', e)
    }

    // Record phase completion (upsert to handle re-completions)
    try {
      await sql`
        INSERT INTO learner_progress (learner_id, fingerprint, day_of_year, phase, time_spent_seconds, year)
        VALUES (${learnerId}, ${fingerprint}, ${dayOfYear}, ${phase}, ${timeSpentSeconds}, ${currentYear})
        ON CONFLICT (fingerprint, day_of_year, phase, year) 
        DO UPDATE SET 
          time_spent_seconds = learner_progress.time_spent_seconds + EXCLUDED.time_spent_seconds,
          completed_at = now()
      `
    } catch (e) {
      console.error('[phase-complete] Failed to record progress:', e)
      return NextResponse.json({ error: 'Failed to save progress' }, { status: 500 })
    }

    // Check if all 5 phases are complete for this day
    const completedPhasesResult = await sql`
      SELECT phase FROM learner_progress 
      WHERE fingerprint = ${fingerprint} 
        AND day_of_year = ${dayOfYear} 
        AND year = ${currentYear}
    `
    const completedPhases = completedPhasesResult.map(r => r.phase as string)
    const dayComplete = VALID_PHASES.every(p => completedPhases.includes(p))

    // If day is complete, update streak
    let currentStreak = 1
    let longestStreak = 1
    let streakUpdated = false

    if (dayComplete && learnerId) {
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const todayStr = today.toISOString().split('T')[0]

      try {
        // Get current learner data
        const learnerData = await sql`
          SELECT current_streak, longest_streak, last_lesson_date 
          FROM learners WHERE id = ${learnerId}
        `
        
        if (learnerData.length > 0) {
          const { current_streak, longest_streak: prev_longest, last_lesson_date } = learnerData[0]
          
          // Calculate if this continues a streak
          const lastDate = last_lesson_date ? new Date(last_lesson_date as string) : null
          lastDate?.setHours(0, 0, 0, 0)
          
          const yesterday = new Date(today)
          yesterday.setDate(yesterday.getDate() - 1)
          
          const isSameDay = lastDate?.getTime() === today.getTime()
          const isConsecutive = lastDate?.getTime() === yesterday.getTime()
          
          if (isSameDay) {
            // Already counted today
            currentStreak = Number(current_streak) || 1
            longestStreak = Number(prev_longest) || 1
          } else if (isConsecutive) {
            // Continuing streak
            currentStreak = (Number(current_streak) || 0) + 1
            longestStreak = Math.max(currentStreak, Number(prev_longest) || 1)
            streakUpdated = true
          } else {
            // New streak (streak was broken or first lesson)
            currentStreak = 1
            longestStreak = Math.max(1, Number(prev_longest) || 1)
            streakUpdated = true
          }

          // Calculate streak bonus XP
          let streakBonus = 0
          if (currentStreak > 0 && currentStreak % 30 === 0) {
            streakBonus = MONTHLY_STREAK_BONUS
          } else if (currentStreak > 0 && currentStreak % 7 === 0) {
            streakBonus = WEEKLY_STREAK_BONUS
          }

          if (streakUpdated || !isSameDay) {
            await sql`
              UPDATE learners 
              SET current_streak = ${currentStreak},
                  longest_streak = ${longestStreak},
                  last_lesson_date = ${todayStr}::date,
                  streak_updated_at = now(),
                  total_xp = COALESCE(total_xp, 0) + ${DAY_COMPLETION_XP + streakBonus}
              WHERE id = ${learnerId}
            `

            // Record in day_completions table
            await sql`
              INSERT INTO day_completions (learner_id, fingerprint, day_of_year, year, phases_completed, xp_earned)
              VALUES (${learnerId}, ${fingerprint}, ${dayOfYear}, ${currentYear}, ${completedPhases}, ${DAY_COMPLETION_XP + streakBonus})
              ON CONFLICT (fingerprint, day_of_year, year) DO NOTHING
            `
          }
        }
      } catch (e) {
        console.error('[phase-complete] Failed to update streak:', e)
      }
    }

    // Calculate XP earned for this action
    const phaseXp = PHASE_XP[phase]
    const totalXpEarned = dayComplete ? (DAY_COMPLETION_XP + phaseXp) : phaseXp

    return NextResponse.json({
      success: true,
      phase,
      dayOfYear,
      completedPhases,
      dayComplete,
      xp: {
        phaseXp,
        dayCompletionXp: dayComplete ? DAY_COMPLETION_XP : 0,
        totalEarned: totalXpEarned
      },
      streak: {
        current: currentStreak,
        longest: longestStreak,
        updated: streakUpdated
      }
    })

  } catch (error) {
    console.error('[phase-complete] Error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

// GET endpoint to retrieve progress for a specific day
export async function GET(request: NextRequest) {
  try {
    const cookieStore = await cookies()
    const fingerprint = cookieStore.get('ck_fp')?.value

    if (!fingerprint) {
      return NextResponse.json({ completedPhases: [], dayComplete: false })
    }

    const { searchParams } = new URL(request.url)
    const dayOfYear = parseInt(searchParams.get('day') || '0', 10)
    const currentYear = new Date().getFullYear()

    if (!dayOfYear) {
      return NextResponse.json({ error: 'Missing day parameter' }, { status: 400 })
    }

    const completedPhasesResult = await sql`
      SELECT phase FROM learner_progress 
      WHERE fingerprint = ${fingerprint} 
        AND day_of_year = ${dayOfYear} 
        AND year = ${currentYear}
    `
    
    const completedPhases = completedPhasesResult.map(r => r.phase as string)
    const dayComplete = VALID_PHASES.every(p => completedPhases.includes(p))

    return NextResponse.json({
      completedPhases,
      dayComplete
    })

  } catch (error) {
    console.error('[phase-complete] GET Error:', error)
    return NextResponse.json({ completedPhases: [], dayComplete: false })
  }
}
