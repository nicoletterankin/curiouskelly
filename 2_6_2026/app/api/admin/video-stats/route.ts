import { NextResponse } from 'next/server'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

/**
 * GET /api/admin/video-stats
 * 
 * Returns comprehensive video production statistics for the admin dashboard.
 */
export async function GET() {
  try {
    const sql = getSql()
    // Get overall stats
    const statsQuery = await sql`
      SELECT 
        COUNT(*) as total_assets,
        COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END) as with_video,
        COUNT(CASE WHEN audio_url IS NOT NULL THEN 1 END) as with_audio,
        COUNT(CASE WHEN status = 'pending' OR status = 'pending_translation' THEN 1 END) as pending,
        COUNT(CASE WHEN status = 'processing' OR status = 'generating' THEN 1 END) as in_progress,
        COUNT(CASE WHEN status = 'failed' OR status = 'error' THEN 1 END) as failed
      FROM kelly_lesson_assets
    `

    // Get day-by-day status
    const dayStatusQuery = await sql`
      SELECT 
        day_number,
        COUNT(*) as total_variants,
        COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END) as with_video,
        COUNT(CASE WHEN audio_url IS NOT NULL THEN 1 END) as with_audio,
        array_agg(DISTINCT phase) as phases,
        array_agg(DISTINCT age_group) as age_groups
      FROM kelly_lesson_assets
      WHERE video_url IS NOT NULL OR audio_url IS NOT NULL
      GROUP BY day_number
      ORDER BY day_number
    `

    // Count days with complete video (all 5 phases)
    const completeDaysQuery = await sql`
      SELECT COUNT(DISTINCT day_number) as complete_days
      FROM (
        SELECT day_number
        FROM kelly_lesson_assets
        WHERE video_url IS NOT NULL
        GROUP BY day_number
        HAVING COUNT(DISTINCT phase) >= 5
      ) sub
    `

    // Try to get HeyGen credits (optional)
    let heygenCredits = null
    try {
      const apiKey = process.env.HEYGEN_API_KEY
      if (apiKey) {
        const res = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
          headers: { 'X-Api-Key': apiKey },
        })
        if (res.ok) {
          const data = await res.json()
          heygenCredits = data?.data?.remaining_quota || null
        }
      }
    } catch {
      // Ignore HeyGen API errors
    }

    const rawStats = statsQuery[0] as Record<string, string>
    const completeDays = completeDaysQuery[0] as Record<string, string>

    return NextResponse.json({
      stats: {
        total_assets: parseInt(rawStats.total_assets || '0'),
        with_video: parseInt(rawStats.with_video || '0'),
        with_audio: parseInt(rawStats.with_audio || '0'),
        pending: parseInt(rawStats.pending || '0'),
        in_progress: parseInt(rawStats.in_progress || '0'),
        failed: parseInt(rawStats.failed || '0'),
        days_complete: parseInt(completeDays.complete_days || '0'),
        heygen_credits: heygenCredits,
      },
      dayStatuses: dayStatusQuery.map(row => {
        const r = row as Record<string, unknown>
        return {
          day_number: r.day_number,
          total_variants: parseInt(r.total_variants as string || '0'),
          with_video: parseInt(r.with_video as string || '0'),
          with_audio: parseInt(r.with_audio as string || '0'),
          phases: r.phases as string[],
          age_groups: r.age_groups as string[],
          completion_pct: Math.round((parseInt(r.with_video as string || '0') / 5) * 100),
        }
      }),
    })
  } catch (error) {
    console.error('[video-stats] Error:', error)
    return NextResponse.json({ 
      error: 'Failed to fetch video stats',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}
