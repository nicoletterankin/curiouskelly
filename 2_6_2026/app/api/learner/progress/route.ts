import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { cookies } from 'next/headers'

export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  try {
    const cookieStore = await cookies()
    const fingerprint = cookieStore.get('ck_fp')?.value
    
    if (!fingerprint) {
      // Return default progress for anonymous users
      const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000)
      return NextResponse.json({
        lessonsCompleted: Math.min(dayOfYear, 22), // Demo: show up to current day
        currentStreak: 1,
        longestStreak: 1,
        totalMinutesLearned: Math.min(dayOfYear, 22) * 5,
        favoriteTopics: ['Science', 'Nature', 'History']
      })
    }

    // Try to get learner progress from database
    try {
      const learnerProgress = await sql`
        SELECT 
          COUNT(DISTINCT lp.day_of_year) as lessons_completed,
          COALESCE(l.current_streak, 1) as current_streak,
          COALESCE(l.longest_streak, 1) as longest_streak,
          COUNT(DISTINCT lp.day_of_year) * 5 as total_minutes
        FROM learners l
        LEFT JOIN learner_progress lp ON l.id = lp.learner_id
        WHERE l.fingerprint = ${fingerprint}
        GROUP BY l.id, l.current_streak, l.longest_streak
        LIMIT 1
      `

      if (learnerProgress.length > 0) {
        const p = learnerProgress[0]
        return NextResponse.json({
          lessonsCompleted: Number(p.lessons_completed) || 1,
          currentStreak: Number(p.current_streak) || 1,
          longestStreak: Number(p.longest_streak) || 1,
          totalMinutesLearned: Number(p.total_minutes) || 5,
          favoriteTopics: ['Science', 'Nature', 'History']
        })
      }
    } catch (dbError) {
      console.warn('[learner/progress] DB query failed, using defaults')
    }

    // Fallback for users without progress records
    const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000)
    return NextResponse.json({
      lessonsCompleted: Math.min(dayOfYear, 22),
      currentStreak: 1,
      longestStreak: 1,
      totalMinutesLearned: Math.min(dayOfYear, 22) * 5,
      favoriteTopics: ['Science', 'Nature', 'History']
    })

  } catch (error) {
    console.error('[learner/progress] Error:', error)
    return NextResponse.json({
      lessonsCompleted: 1,
      currentStreak: 1,
      longestStreak: 1,
      totalMinutesLearned: 5,
      favoriteTopics: []
    })
  }
}
