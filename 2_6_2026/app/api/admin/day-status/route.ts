/**
 * Quick status check for any day's content pipeline
 * 
 * GET /api/admin/day-status?day=34
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'

export async function GET(request: NextRequest) {
  const sql = getSql()
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '34')

  try {
    // Get lesson info
    const lesson = await sql`
      SELECT day_of_year, title, topic, theme, hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lessons WHERE day_of_year = ${day}
    `

    // Get asset counts
    const assets = await sql`
      SELECT 
        COUNT(*) as total,
        COUNT(audio_url) as with_audio,
        COUNT(video_url) as with_video,
        COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
      FROM kelly_lesson_assets WHERE day_number = ${day}
    `

    // Get HeyGen video counts
    const heygen = await sql`
      SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
      FROM heygen_videos WHERE day_of_year = ${day}
    `

    // Sample scripts to verify content
    const sampleAssets = await sql`
      SELECT phase, script_text, audio_url IS NOT NULL as has_audio
      FROM kelly_lesson_assets 
      WHERE day_number = ${day} AND language = 'en' AND age_group = 'adult'
      ORDER BY phase
      LIMIT 5
    `

    return NextResponse.json({
      day,
      lesson: lesson[0] || null,
      assets: assets[0],
      heygen: heygen[0],
      sampleAssets,
      readyForProduction: lesson[0]?.title && (assets[0]?.total || 0) > 0,
      nextSteps: getNextSteps(lesson[0], assets[0], heygen[0])
    })
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}

function getNextSteps(lesson: Record<string, unknown> | null, assets: Record<string, unknown> | null, heygen: Record<string, unknown> | null) {
  const steps = []
  
  if (!lesson) {
    steps.push('Missing lesson entry in database')
    return steps
  }
  
  if (!assets || Number(assets.total) === 0) {
    steps.push('Need to create kelly_lesson_assets entries')
    return steps
  }
  
  if (Number(assets.with_audio) === 0) {
    steps.push('Audio will generate on-demand when users visit the page')
    steps.push('Or call POST /api/audio/batch with { days: [34] } to pre-generate')
  }
  
  if (Number(assets.with_video) === 0) {
    steps.push('After audio exists, trigger HeyGen via POST /api/heygen/pipeline')
  }
  
  if (steps.length === 0) {
    steps.push('All content ready!')
  }
  
  return steps
}
