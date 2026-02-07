/**
 * EXECUTIVE COMMAND CENTER API
 * 
 * Single source of truth for pipeline status, coverage gaps,
 * and one-click orchestration of the full video production pipeline.
 * 
 * GET /api/command-center
 *   Returns: Full executive dashboard data -- coverage by day, phase,
 *   age, archetype; gap analysis; estimated costs & time; action items.
 * 
 * POST /api/command-center
 *   Body: { action: "sync-lessons" }          -> Sync missing lessons from Supabase
 *   Body: { action: "seed-all" }              -> Seed all 180 slots for all days with lessons
 *   Body: { action: "generate-audio", ... }   -> Generate ElevenLabs audio for slots missing it
 *   Body: { action: "generate-video", ... }   -> Generate Fal.ai lip-sync video
 *   Body: { action: "full-day", day: N }      -> Run complete pipeline for one day
 *   Body: { action: "full-range", start, end }-> Run complete pipeline for a range
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { BASE_VIDEOS } from '@/lib/fal-lipsync-client'
import {
  PIPELINE_AGES,
  PIPELINE_ARCHETYPES,
  PIPELINE_PHASES,
  SLOTS_PER_DAY,
  TOTAL_SLOTS,
  TOTAL_DAYS,
} from '@/lib/production-pipeline'

// ============================================================================
// GET - Executive Dashboard Data
// ============================================================================

export async function GET() {
  const startTime = Date.now()

  try {
    const sql = getSql()

    // 1. Lesson content coverage
    const lessonStats = await sql`
      SELECT 
        COUNT(DISTINCT day_of_year)::int as total_days,
        COUNT(*) FILTER (WHERE track = 'learn')::int as learn_count,
        COUNT(*) FILTER (WHERE track = 'grow')::int as grow_count,
        COUNT(*) FILTER (WHERE hook_script IS NOT NULL)::int as with_hook,
        COUNT(*) FILTER (WHERE story_script IS NOT NULL)::int as with_story,
        COUNT(*) FILTER (WHERE wonder_script IS NOT NULL)::int as with_wonder,
        COUNT(*) FILTER (WHERE action_script IS NOT NULL)::int as with_action,
        COUNT(*) FILTER (WHERE wisdom_script IS NOT NULL)::int as with_wisdom,
        MIN(day_of_year)::int as min_day,
        MAX(day_of_year)::int as max_day
      FROM lessons
    `
    const ls = lessonStats[0] || {}

    // Find missing days
    const existingDays = await sql`
      SELECT DISTINCT day_of_year FROM lessons ORDER BY day_of_year
    `
    const existingDaySet = new Set((existingDays as Array<{ day_of_year: number }>).map(d => d.day_of_year))
    const missingDays: number[] = []
    for (let d = 1; d <= 365; d++) {
      if (!existingDaySet.has(d)) missingDays.push(d)
    }

    // Days with full scripts (all 5 phases)
    const daysWithAllScripts = await sql`
      SELECT COUNT(*)::int as count FROM lessons
      WHERE hook_script IS NOT NULL 
        AND story_script IS NOT NULL
        AND wonder_script IS NOT NULL
        AND action_script IS NOT NULL
        AND wisdom_script IS NOT NULL
    `

    // 2. kelly_lesson_assets pipeline coverage
    const assetStats = await sql`
      SELECT 
        COUNT(*)::int as total_rows,
        COUNT(audio_url)::int as with_audio,
        COUNT(video_url)::int as with_video,
        COUNT(script_text)::int as with_script,
        COUNT(DISTINCT day_number)::int as days_seeded,
        COUNT(*) FILTER (WHERE status = 'completed')::int as status_completed,
        COUNT(*) FILTER (WHERE status = 'audio_ready')::int as status_audio_ready,
        COUNT(*) FILTER (WHERE status = 'pending')::int as status_pending
      FROM kelly_lesson_assets
      WHERE language = 'en'
    `
    const as_ = assetStats[0] || {}

    // 3. Coverage by age
    const byAge = await sql`
      SELECT 
        age_group,
        COUNT(*)::int as total,
        COUNT(audio_url)::int as audio,
        COUNT(video_url)::int as video
      FROM kelly_lesson_assets
      WHERE language = 'en'
      GROUP BY age_group
      ORDER BY age_group
    `

    // 4. Coverage by phase
    const byPhase = await sql`
      SELECT 
        phase,
        COUNT(*)::int as total,
        COUNT(audio_url)::int as audio,
        COUNT(video_url)::int as video
      FROM kelly_lesson_assets
      WHERE language = 'en'
      GROUP BY phase
      ORDER BY 
        CASE phase 
          WHEN 'hook' THEN 1 WHEN 'story' THEN 2 WHEN 'wonder' THEN 3 
          WHEN 'action' THEN 4 WHEN 'wisdom' THEN 5 
        END
    `

    // 5. Coverage by archetype
    const byArchetype = await sql`
      SELECT 
        archetype,
        COUNT(*)::int as total,
        COUNT(audio_url)::int as audio,
        COUNT(video_url)::int as video
      FROM kelly_lesson_assets
      WHERE language = 'en'
      GROUP BY archetype
      ORDER BY archetype
    `

    // 6. Day-level heatmap data (first 365 days)
    const dayHeatmap = await sql`
      SELECT 
        day_number,
        COUNT(*)::int as seeded,
        COUNT(audio_url)::int as audio,
        COUNT(video_url)::int as video
      FROM kelly_lesson_assets
      WHERE language = 'en'
      GROUP BY day_number
      ORDER BY day_number
    `

    // 7. Base video coverage
    const baseVideoCount = Object.keys(BASE_VIDEOS).length
    const baseVideoMissing = PIPELINE_AGES.flatMap(age =>
      PIPELINE_ARCHETYPES
        .filter(arch => !BASE_VIDEOS[`${age}_${arch}`])
        .map(arch => `${age}_${arch}`)
    )

    // 8. Lesson audio table stats (legacy)
    const legacyAudio = await sql`
      SELECT 
        COUNT(*)::int as total,
        COUNT(audio_url)::int as with_audio,
        COUNT(DISTINCT day_number)::int as days
      FROM lesson_audio
    `.catch(() => [{ total: 0, with_audio: 0, days: 0 }])

    // 9. Cost estimates
    const totalWithAudio = parseInt(as_.with_audio as string) || 0
    const totalWithVideo = parseInt(as_.with_video as string) || 0
    const totalSeeded = parseInt(as_.total_rows as string) || 0

    const audioNeeded = TOTAL_SLOTS - totalWithAudio
    const videoNeeded = TOTAL_SLOTS - totalWithVideo
    const slotsToSeed = TOTAL_SLOTS - totalSeeded

    // ElevenLabs: ~$0.30 per 1000 chars, avg script ~400 chars = ~$0.12 per audio
    const audioCostEstimate = audioNeeded * 0.12
    // Fal.ai: ~$0.04 per lip-sync call
    const videoCostEstimate = videoNeeded * 0.04
    // Vercel Blob: ~$0.03/GB stored, avg video 3MB = ~$0.00009/video stored
    const storageCostEstimate = videoNeeded * 0.00009 + audioNeeded * 0.000003

    // Time estimates (processing)
    // ElevenLabs: ~2s per audio clip + rate limits
    const audioTimeHours = audioNeeded * 2 / 3600
    // Fal.ai: ~15s per lip-sync + parallel batches of 5
    const videoTimeHours = (videoNeeded * 15 / 5) / 3600

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      elapsed_seconds: elapsed,

      // ====================== EXECUTIVE SUMMARY ======================
      executive: {
        pipeline_health: totalWithVideo > 0 ? 'operational' : 'cold_start',
        days_with_lessons: parseInt(ls.total_days as string) || 0,
        days_missing_lessons: missingDays.length,
        days_with_full_scripts: parseInt(daysWithAllScripts[0]?.count as string) || 0,
        base_videos: `${baseVideoCount}/36`,
        base_videos_complete: baseVideoCount === 36,
        total_slots_needed: TOTAL_SLOTS,
        slots_seeded: totalSeeded,
        slots_with_audio: totalWithAudio,
        slots_with_video: totalWithVideo,
        audio_coverage_pct: parseFloat(((totalWithAudio / TOTAL_SLOTS) * 100).toFixed(2)),
        video_coverage_pct: parseFloat(((totalWithVideo / TOTAL_SLOTS) * 100).toFixed(2)),
      },

      // ====================== GAP ANALYSIS ======================
      gaps: {
        missing_lesson_days: {
          count: missingDays.length,
          days: missingDays.slice(0, 60),
          has_more: missingDays.length > 60,
          action: 'POST /api/command-center { action: "sync-lessons" }',
        },
        slots_to_seed: {
          count: slotsToSeed,
          action: 'POST /api/command-center { action: "seed-all" }',
        },
        audio_to_generate: {
          count: audioNeeded,
          action: 'POST /api/command-center { action: "generate-audio", dayStart: 1, dayEnd: 365, limit: 100 }',
        },
        video_to_generate: {
          count: videoNeeded,
          action: 'POST /api/command-center { action: "generate-video", dayStart: 1, dayEnd: 365, limit: 50 }',
        },
        base_video_gaps: {
          count: baseVideoMissing.length,
          missing: baseVideoMissing,
        },
      },

      // ====================== COST & TIME ESTIMATES ======================
      estimates: {
        audio: {
          remaining: audioNeeded,
          cost_usd: parseFloat(audioCostEstimate.toFixed(2)),
          time_hours: parseFloat(audioTimeHours.toFixed(1)),
          note: 'ElevenLabs TTS at ~$0.12/clip avg, ~2s/clip with rate limits',
        },
        video: {
          remaining: videoNeeded,
          cost_usd: parseFloat(videoCostEstimate.toFixed(2)),
          time_hours: parseFloat(videoTimeHours.toFixed(1)),
          note: 'Fal.ai lip-sync at ~$0.04/clip, ~15s/clip in parallel batches of 5',
        },
        storage: {
          cost_usd: parseFloat(storageCostEstimate.toFixed(2)),
          note: 'Vercel Blob storage for audio + video files',
        },
        total_cost_usd: parseFloat((audioCostEstimate + videoCostEstimate + storageCostEstimate).toFixed(2)),
        total_time_hours: parseFloat((audioTimeHours + videoTimeHours).toFixed(1)),
      },

      // ====================== BREAKDOWNS ======================
      breakdowns: {
        by_age: byAge,
        by_phase: byPhase,
        by_archetype: byArchetype,
      },

      // ====================== DAY HEATMAP ======================
      heatmap: dayHeatmap,

      // ====================== CONFIGURATION ======================
      config: {
        ages: [...PIPELINE_AGES],
        archetypes: [...PIPELINE_ARCHETYPES],
        phases: [...PIPELINE_PHASES],
        slots_per_day: SLOTS_PER_DAY,
        total_days: TOTAL_DAYS,
        total_slots: TOTAL_SLOTS,
        languages: ['en'],
        video_dimensions: '1920x1080',
      },

      // ====================== LEGACY TABLES ======================
      legacy: {
        lesson_audio: legacyAudio[0] || {},
      },

      // ====================== RECOMMENDED ACTIONS (ORDERED) ======================
      actions: [
        ...(missingDays.length > 0 ? [{
          priority: 1,
          action: 'sync-lessons',
          description: `Sync ${missingDays.length} missing lesson days from Supabase`,
          endpoint: 'POST /api/command-center',
          body: { action: 'sync-lessons' },
        }] : []),
        ...(slotsToSeed > 0 ? [{
          priority: 2,
          action: 'seed-all',
          description: `Seed ${slotsToSeed.toLocaleString()} missing kelly_lesson_assets rows`,
          endpoint: 'POST /api/command-center',
          body: { action: 'seed-all' },
        }] : []),
        ...(audioNeeded > 0 ? [{
          priority: 3,
          action: 'generate-audio',
          description: `Generate ${audioNeeded.toLocaleString()} audio clips via ElevenLabs (~$${audioCostEstimate.toFixed(0)})`,
          endpoint: 'POST /api/command-center',
          body: { action: 'generate-audio', dayStart: 1, dayEnd: 365, limit: 100 },
        }] : []),
        ...(videoNeeded > 0 ? [{
          priority: 4,
          action: 'generate-video',
          description: `Generate ${videoNeeded.toLocaleString()} lip-sync videos via Fal.ai (~$${videoCostEstimate.toFixed(0)})`,
          endpoint: 'POST /api/command-center',
          body: { action: 'generate-video', dayStart: 1, dayEnd: 365, limit: 50 },
        }] : []),
      ],
    })
  } catch (error) {
    console.error('[command-center] GET error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

// ============================================================================
// POST - Execute Pipeline Actions
// ============================================================================

interface CommandAction {
  action: 'sync-lessons' | 'seed-all' | 'seed-day' | 'generate-audio' | 'generate-video' | 'full-day' | 'full-range'
  day?: number
  dayStart?: number
  dayEnd?: number
  limit?: number
  dryRun?: boolean
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const body = await request.json() as CommandAction
    const { action, day, dayStart = 1, dayEnd = 365, limit = 100, dryRun = false } = body

    console.log(`[command-center] Executing action: ${action}`, { dayStart, dayEnd, limit, dryRun })

    switch (action) {
      // ====================================================================
      // SYNC LESSONS: Fill missing days from Supabase
      // ====================================================================
      case 'sync-lessons': {
        const syncRes = await fetch(new URL('/api/sync/core-lessons', request.url), {
          method: 'POST',
        })
        const syncData = await syncRes.json()
        return NextResponse.json({
          success: true,
          action: 'sync-lessons',
          result: syncData,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // SEED ALL: Ensure kelly_lesson_assets has all 180 rows per day
      // ====================================================================
      case 'seed-all': {
        const { seedDayRange } = await import('@/lib/production-pipeline')
        const sql = getSql()
        const result = await seedDayRange(sql, dayStart, dayEnd)
        return NextResponse.json({
          success: true,
          action: 'seed-all',
          range: { dayStart, dayEnd },
          ...result,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // SEED DAY: Seed a single day
      // ====================================================================
      case 'seed-day': {
        if (!day) {
          return NextResponse.json({ error: 'day is required' }, { status: 400 })
        }
        const { seedDaySlots } = await import('@/lib/production-pipeline')
        const sql = getSql()
        const result = await seedDaySlots(sql, day)
        return NextResponse.json({
          success: true,
          action: 'seed-day',
          day,
          ...result,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // GENERATE AUDIO: Call full-coverage pipeline audio generator
      // ====================================================================
      case 'generate-audio': {
        const pipelineRes = await fetch(new URL('/api/pipeline/full-coverage', request.url), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'generate-audio',
            dayStart,
            dayEnd,
            limit,
            dryRun,
          }),
        })
        const pipelineData = await pipelineRes.json()
        return NextResponse.json({
          success: true,
          action: 'generate-audio',
          result: pipelineData,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // GENERATE VIDEO: Call full-coverage pipeline video generator
      // ====================================================================
      case 'generate-video': {
        const pipelineRes = await fetch(new URL('/api/pipeline/full-coverage', request.url), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'generate-video',
            dayStart,
            dayEnd,
            limit,
            dryRun,
          }),
        })
        const pipelineData = await pipelineRes.json()
        return NextResponse.json({
          success: true,
          action: 'generate-video',
          result: pipelineData,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // FULL DAY: Seed + Audio + Video for a single day
      // ====================================================================
      case 'full-day': {
        if (!day) {
          return NextResponse.json({ error: 'day is required' }, { status: 400 })
        }
        const pipelineRes = await fetch(new URL('/api/pipeline/full-coverage', request.url), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'full-day', day, dryRun }),
        })
        const pipelineData = await pipelineRes.json()
        return NextResponse.json({
          success: true,
          action: 'full-day',
          day,
          result: pipelineData,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // FULL RANGE: Seed + Audio + Video for a range of days
      // ====================================================================
      case 'full-range': {
        const steps: Array<{ step: string; result: unknown }> = []

        // Step 1: Seed all slots
        const seedRes = await fetch(new URL('/api/command-center', request.url), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'seed-all', dayStart, dayEnd }),
        })
        steps.push({ step: 'seed', result: await seedRes.json() })

        if (!dryRun) {
          // Step 2: Generate audio
          const audioRes = await fetch(new URL('/api/command-center', request.url), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'generate-audio', dayStart, dayEnd, limit }),
          })
          steps.push({ step: 'audio', result: await audioRes.json() })

          // Step 3: Generate video
          const videoRes = await fetch(new URL('/api/command-center', request.url), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'generate-video', dayStart, dayEnd, limit }),
          })
          steps.push({ step: 'video', result: await videoRes.json() })
        }

        return NextResponse.json({
          success: true,
          action: 'full-range',
          range: { dayStart, dayEnd },
          steps,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        )
    }
  } catch (error) {
    console.error('[command-center] POST error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
