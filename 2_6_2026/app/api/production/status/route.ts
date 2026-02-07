/**
 * Production Status API
 * Returns comprehensive coverage statistics for lesson assets
 */

import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

export async function GET() {
  try {
    // Get overall kelly_lesson_assets stats
    const kellyStats = await sql`
      SELECT 
        COUNT(DISTINCT day_number) as days_covered,
        COUNT(*) as total_rows,
        COUNT(audio_url) as audio_count,
        COUNT(video_url) as video_count,
        COUNT(DISTINCT age_group) as age_groups,
        COUNT(DISTINCT phase) as phases
      FROM kelly_lesson_assets
      WHERE audio_url IS NOT NULL OR video_url IS NOT NULL
    `

    // Get coverage by age group
    const ageGroupStats = await sql`
      SELECT 
        age_group,
        COUNT(DISTINCT day_number) as days_covered,
        COUNT(*) as total_phases,
        COUNT(audio_url) as with_audio,
        COUNT(video_url) as with_video
      FROM kelly_lesson_assets
      GROUP BY age_group
      ORDER BY 
        CASE age_group 
          WHEN 'toddler' THEN 1 
          WHEN 'preteen' THEN 2 
          WHEN 'youngAdult' THEN 3 
          WHEN 'middleAge' THEN 4 
          WHEN 'senior' THEN 5 
          ELSE 6 
        END
    `

    // Get coverage by phase
    const phaseStats = await sql`
      SELECT 
        phase,
        COUNT(DISTINCT day_number) as days_covered,
        COUNT(audio_url) as with_audio,
        COUNT(video_url) as with_video
      FROM kelly_lesson_assets
      GROUP BY phase
      ORDER BY 
        CASE phase 
          WHEN 'hook' THEN 1 
          WHEN 'story' THEN 2 
          WHEN 'wonder' THEN 3 
          WHEN 'action' THEN 4 
          WHEN 'wisdom' THEN 5 
        END
    `

    // Get heygen_videos stats
    const heygenStats = await sql`
      SELECT 
        COUNT(*) FILTER (WHERE status IN ('completed', 'placeholder', 'ready')) as completed,
        COUNT(*) FILTER (WHERE status = 'queued') as queued,
        COUNT(*) FILTER (WHERE status = 'failed') as failed,
        COUNT(DISTINCT day_of_year) as days_covered
      FROM heygen_videos
    `

    // Get generated_assets stats
    const generatedStats = await sql`
      SELECT 
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE asset_type = 'audio') as audio,
        COUNT(*) FILTER (WHERE asset_type = 'video') as video
      FROM generated_assets
      WHERE status = 'completed'
    `

    // Calculate completeness percentages
    const totalExpected = 365 * 5 // days × phases per age group
    const kelly = kellyStats[0] || {}
    const heygen = heygenStats[0] || {}
    const generated = generatedStats[0] || {}

    // Format for dashboard consumption
    const summary = {
      kellyLessonAssets: {
        total: Number(kelly.total_rows) || 0,
        withAudio: Number(kelly.audio_count) || 0,
        withVideo: Number(kelly.video_count) || 0,
        daysCovered: Number(kelly.days_covered) || 0,
      },
      heygenVideos: {
        completed: Number(heygen.completed) || 0,
        queued: Number(heygen.queued) || 0,
        failed: Number(heygen.failed) || 0,
        daysCovered: Number(heygen.days_covered) || 0,
      },
      generatedAssets: {
        total: Number(generated.total) || 0,
        audio: Number(generated.audio) || 0,
        video: Number(generated.video) || 0,
      },
      coverage: {
        audioPercent: ((Number(kelly.audio_count) || 0) / totalExpected) * 100,
        videoPercent: ((Number(heygen.completed) || 0) / totalExpected) * 100,
        totalLessons: 365,
      },
      byAgeGroup: ageGroupStats,
      byPhase: phaseStats,
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json(summary)
  } catch (error) {
    console.error('[production/status] Error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch production status' },
      { status: 500 }
    )
  }
}
