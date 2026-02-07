/**
 * FULL COVERAGE PIPELINE API
 * 
 * Single endpoint to understand and drive complete video coverage.
 * 
 * GET /api/pipeline/full-coverage
 *   Returns pipeline summary: how many days have lessons, how many slots seeded,
 *   how many have audio, how many have video, broken down by age/archetype/phase.
 * 
 * GET /api/pipeline/full-coverage?day=42
 *   Returns detailed coverage for a specific day.
 * 
 * GET /api/pipeline/full-coverage?dayStart=1&dayEnd=30
 *   Returns coverage for a range of days.
 * 
 * POST /api/pipeline/full-coverage
 *   Body: { action: "seed", dayStart: 1, dayEnd: 30 }
 *     -> Seeds all 180 slots per day into kelly_lesson_assets for days that have lessons
 *   
 *   Body: { action: "generate-audio", dayStart: 1, dayEnd: 30, limit: 50 }
 *     -> Generates ElevenLabs audio for seeded slots missing audio
 *   
 *   Body: { action: "generate-video", dayStart: 1, dayEnd: 30, limit: 25 }
 *     -> Generates Fal.ai lip-sync video for slots with audio but no video
 *   
 *   Body: { action: "full-day", day: 42 }
 *     -> Runs the full pipeline for a single day: seed -> audio -> video
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'
import { generateLipSyncVideo, getBaseVideo } from '@/lib/fal-lipsync-client'
import {
  getPipelineSummary,
  getDayCoverage,
  getDayRangeCoverage,
  seedDaySlots,
  seedDayRange,
  findSlotsNeedingAudio,
  findSlotsNeedingVideo,
  PIPELINE_PHASES,
  PHASE_SCRIPT_COLUMNS,
  mapToBaseAge,
  type PipelineAge,
} from '@/lib/production-pipeline'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || ''
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

// Voice IDs per age for distinct audio
const AGE_VOICE_IDS: Record<PipelineAge, string> = {
  kid: process.env.ELEVENLABS_VOICE_KID || ELEVENLABS_VOICE_ID,
  adult: process.env.ELEVENLABS_VOICE_ADULT || ELEVENLABS_VOICE_ID,
  senior: process.env.ELEVENLABS_VOICE_SENIOR || ELEVENLABS_VOICE_ID,
}

// ============================================================================
// GET - Pipeline Status
// ============================================================================

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = searchParams.get('day')
  const dayStart = searchParams.get('dayStart')
  const dayEnd = searchParams.get('dayEnd')

  try {
    const sql = getSql()

    // Specific day detail
    if (day) {
      const coverage = await getDayCoverage(sql, parseInt(day))
      return NextResponse.json({ success: true, coverage })
    }

    // Day range
    if (dayStart && dayEnd) {
      const coverage = await getDayRangeCoverage(sql, parseInt(dayStart), parseInt(dayEnd))
      return NextResponse.json({ success: true, coverage })
    }

    // Full summary
    const summary = await getPipelineSummary(sql)
    return NextResponse.json({ success: true, summary })

  } catch (error) {
    console.error('[pipeline/full-coverage] GET error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

// ============================================================================
// POST - Pipeline Actions
// ============================================================================

interface PipelineAction {
  action: 'seed' | 'generate-audio' | 'generate-video' | 'full-day'
  day?: number
  dayStart?: number
  dayEnd?: number
  age?: PipelineAge
  limit?: number
  dryRun?: boolean
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const body = await request.json() as PipelineAction
    const { action, day, dayStart = 1, dayEnd = 30, age, limit = 50, dryRun = false } = body

    const sql = getSql()

    switch (action) {
      // ====================================================================
      // SEED: Ensure all 180 rows exist for each day
      // ====================================================================
      case 'seed': {
        if (day) {
          const result = await seedDaySlots(sql, day)
          return NextResponse.json({ success: true, action: 'seed', day, ...result })
        }
        const result = await seedDayRange(sql, dayStart, dayEnd)
        return NextResponse.json({ success: true, action: 'seed', dayStart, dayEnd, ...result })
      }

      // ====================================================================
      // GENERATE-AUDIO: Create ElevenLabs audio for slots missing it
      // ====================================================================
      case 'generate-audio': {
        const slots = await findSlotsNeedingAudio(sql, { dayStart, dayEnd, age, limit })

        if (slots.length === 0) {
          return NextResponse.json({
            success: true, action: 'generate-audio',
            message: 'No slots need audio in this range',
            processed: 0,
          })
        }

        if (dryRun) {
          return NextResponse.json({
            success: true, action: 'generate-audio', dryRun: true,
            slotsNeedingAudio: slots.length,
            sample: slots.slice(0, 10),
          })
        }

        // Get lesson scripts for the days we need
        const dayNumbers = [...new Set(slots.map(s => s.day_number))]
        const lessons = await sql`
          SELECT day_of_year, hook_script, story_script, wonder_script, action_script, wisdom_script
          FROM lessons WHERE day_of_year = ANY(${dayNumbers})
        `
        const lessonMap = new Map<number, Record<string, string | null>>()
        for (const l of lessons as Array<Record<string, unknown>>) {
          lessonMap.set(l.day_of_year as number, l as Record<string, string | null>)
        }

        let succeeded = 0
        let failed = 0
        const errors: Array<{ slot: string; error: string }> = []

        for (const slot of slots) {
          try {
            const lesson = lessonMap.get(slot.day_number)
            if (!lesson) {
              errors.push({ slot: `d${slot.day_number}-${slot.phase}`, error: 'No lesson' })
              failed++
              continue
            }

            const scriptCol = PHASE_SCRIPT_COLUMNS[slot.phase as keyof typeof PHASE_SCRIPT_COLUMNS]
            const script = lesson[scriptCol]
            if (!script) {
              errors.push({ slot: `d${slot.day_number}-${slot.phase}`, error: 'No script' })
              failed++
              continue
            }

            // Generate audio via ElevenLabs
            const voiceId = AGE_VOICE_IDS[slot.age_group as PipelineAge] || ELEVENLABS_VOICE_ID
            const ttsResponse = await fetch(
              `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
              {
                method: 'POST',
                headers: {
                  'xi-api-key': ELEVENLABS_API_KEY,
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  text: script,
                  model_id: 'eleven_multilingual_v2',
                  voice_settings: { stability: 0.5, similarity_boost: 0.75 },
                }),
              }
            )

            if (!ttsResponse.ok) {
              const errText = await ttsResponse.text()
              if (ttsResponse.status === 429) {
                // Rate limited - stop processing
                return NextResponse.json({
                  success: true, action: 'generate-audio',
                  rateLimited: true,
                  succeeded, failed,
                  message: 'Rate limited by ElevenLabs, try again later',
                  errors: errors.slice(0, 10),
                  elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
                })
              }
              errors.push({ slot: `d${slot.day_number}-${slot.phase}-${slot.age_group}`, error: errText.slice(0, 100) })
              failed++
              continue
            }

            // Upload to Vercel Blob
            const audioBuffer = await ttsResponse.arrayBuffer()
            const blobPath = `audio/day${String(slot.day_number).padStart(3, '0')}/${slot.phase}_${slot.age_group}_${slot.archetype}.mp3`
            const blob = await put(blobPath, Buffer.from(audioBuffer), {
              access: 'public',
              contentType: 'audio/mpeg',
            })

            // Update kelly_lesson_assets directly
            await sql`
              UPDATE kelly_lesson_assets
              SET audio_url = ${blob.url},
                  script_text = ${script},
                  status = 'audio_ready',
                  updated_at = NOW()
              WHERE id = ${slot.id}
            `

            succeeded++
          } catch (err) {
            errors.push({
              slot: `d${slot.day_number}-${slot.phase}-${slot.age_group}-${slot.archetype}`,
              error: err instanceof Error ? err.message : 'Unknown',
            })
            failed++
          }
        }

        console.log(`[pipeline/full-coverage] Audio batch complete: ${succeeded} succeeded, ${failed} failed out of ${slots.length}`)
        return NextResponse.json({
          success: true, action: 'generate-audio',
          succeeded, failed,
          total: slots.length,
          errors: errors.slice(0, 20),
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // GENERATE-VIDEO: Create Fal.ai lip-sync videos for slots with audio
      // ====================================================================
      case 'generate-video': {
        const slots = await findSlotsNeedingVideo(sql, { dayStart, dayEnd, age, limit })

        if (slots.length === 0) {
          return NextResponse.json({
            success: true, action: 'generate-video',
            message: 'No slots need video in this range',
            processed: 0,
          })
        }

        if (dryRun) {
          return NextResponse.json({
            success: true, action: 'generate-video', dryRun: true,
            slotsNeedingVideo: slots.length,
            sample: slots.slice(0, 10),
          })
        }

        let succeeded = 0
        let failed = 0
        const errors: Array<{ slot: string; error: string }> = []

        // Process in parallel batches of 5
        for (let i = 0; i < slots.length; i += 5) {
          const batch = slots.slice(i, i + 5)
          const results = await Promise.allSettled(
            batch.map(async (slot) => {
              const baseAge = mapToBaseAge(slot.age_group)
              const baseVideoUrl = getBaseVideo(baseAge, slot.archetype)

              if (!baseVideoUrl) {
                throw new Error(`No base video for ${baseAge}/${slot.archetype}`)
              }

              // Generate lip-synced video
              const result = await generateLipSyncVideo({
                videoUrl: baseVideoUrl,
                audioUrl: slot.audio_url,
                syncMode: 'cut',
              })

              // Upload to Vercel Blob
              const videoResponse = await fetch(result.videoUrl)
              const videoBuffer = await videoResponse.arrayBuffer()
              const blobPath = `video/kelly/day-${String(slot.day_number).padStart(3, '0')}/${slot.phase}-${slot.age_group}-${slot.archetype}.mp4`
              const blob = await put(blobPath, videoBuffer, {
                access: 'public',
                contentType: 'video/mp4',
                addRandomSuffix: false,
              })

              // Update database
              await sql`
                UPDATE kelly_lesson_assets
                SET video_url = ${blob.url},
                    video_source = 'fal_lipsync',
                    status = 'completed',
                    updated_at = NOW()
                WHERE id = ${slot.id}
              `

              return { id: slot.id, videoUrl: blob.url }
            })
          )

          for (let j = 0; j < results.length; j++) {
            const result = results[j]
            if (result.status === 'fulfilled') {
              succeeded++
            } else {
              failed++
              errors.push({
                slot: `d${batch[j].day_number}-${batch[j].phase}-${batch[j].age_group}-${batch[j].archetype}`,
                error: result.reason?.message || 'Unknown error',
              })
            }
          }
        }

        console.log(`[pipeline/full-coverage] Video batch complete: ${succeeded} succeeded, ${failed} failed out of ${slots.length}`)
        return NextResponse.json({
          success: true, action: 'generate-video',
          succeeded, failed,
          total: slots.length,
          errors: errors.slice(0, 20),
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // FULL-DAY: Run complete pipeline for a single day
      // ====================================================================
      case 'full-day': {
        if (!day) {
          return NextResponse.json({ success: false, error: 'day is required for full-day action' }, { status: 400 })
        }

        const steps: Array<{ step: string; result: unknown }> = []

        // Step 1: Seed slots
        const seedResult = await seedDaySlots(sql, day)
        steps.push({ step: 'seed', result: seedResult })

        // Step 2: Generate audio (all 180 slots)
        if (!dryRun) {
          const audioSlots = await findSlotsNeedingAudio(sql, { dayStart: day, dayEnd: day, limit: 180 })
          
          if (audioSlots.length > 0) {
            // Get lesson scripts
            const lessons = await sql`
              SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
              FROM lessons WHERE day_of_year = ${day} LIMIT 1
            `
            const lesson = lessons[0] as Record<string, string | null> | undefined

            let audioSucceeded = 0
            let audioFailed = 0

            if (lesson) {
              for (const slot of audioSlots) {
                try {
                  const scriptCol = PHASE_SCRIPT_COLUMNS[slot.phase as keyof typeof PHASE_SCRIPT_COLUMNS]
                  const script = lesson[scriptCol]
                  if (!script) { audioFailed++; continue }

                  const voiceId = AGE_VOICE_IDS[slot.age_group as PipelineAge] || ELEVENLABS_VOICE_ID
                  const ttsResponse = await fetch(
                    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
                    {
                      method: 'POST',
                      headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        text: script,
                        model_id: 'eleven_multilingual_v2',
                        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
                      }),
                    }
                  )

                  if (ttsResponse.status === 429) {
                    steps.push({ step: 'audio', result: { audioSucceeded, audioFailed, rateLimited: true } })
                    break
                  }

                  if (!ttsResponse.ok) { audioFailed++; continue }

                  const audioBuffer = await ttsResponse.arrayBuffer()
                  const blobPath = `audio/day${String(day).padStart(3, '0')}/${slot.phase}_${slot.age_group}_${slot.archetype}.mp3`
                  const blob = await put(blobPath, Buffer.from(audioBuffer), {
                    access: 'public', contentType: 'audio/mpeg',
                  })

                  await sql`
                    UPDATE kelly_lesson_assets
                    SET audio_url = ${blob.url}, script_text = ${script}, status = 'audio_ready', updated_at = NOW()
                    WHERE id = ${slot.id}
                  `
                  audioSucceeded++
                } catch {
                  audioFailed++
                }
              }
            }
            steps.push({ step: 'audio', result: { audioSucceeded, audioFailed, total: audioSlots.length } })
          } else {
            steps.push({ step: 'audio', result: { message: 'All slots already have audio' } })
          }

          // Step 3: Generate videos
          const videoSlots = await findSlotsNeedingVideo(sql, { dayStart: day, dayEnd: day, limit: 180 })
          
          if (videoSlots.length > 0) {
            let videoSucceeded = 0
            let videoFailed = 0

            for (let i = 0; i < videoSlots.length; i += 5) {
              const batch = videoSlots.slice(i, i + 5)
              const results = await Promise.allSettled(
                batch.map(async (slot) => {
                  const baseAge = mapToBaseAge(slot.age_group)
                  const baseVideoUrl = getBaseVideo(baseAge, slot.archetype)
                  if (!baseVideoUrl) throw new Error('No base video')

                  const result = await generateLipSyncVideo({
                    videoUrl: baseVideoUrl, audioUrl: slot.audio_url, syncMode: 'cut',
                  })

                  const videoResponse = await fetch(result.videoUrl)
                  const videoBuffer = await videoResponse.arrayBuffer()
                  const blobPath = `video/kelly/day-${String(day).padStart(3, '0')}/${slot.phase}-${slot.age_group}-${slot.archetype}.mp4`
                  const blob = await put(blobPath, videoBuffer, {
                    access: 'public', contentType: 'video/mp4', addRandomSuffix: false,
                  })

                  await sql`
                    UPDATE kelly_lesson_assets
                    SET video_url = ${blob.url}, video_source = 'fal_lipsync', status = 'completed', updated_at = NOW()
                    WHERE id = ${slot.id}
                  `
                })
              )

              for (const r of results) {
                if (r.status === 'fulfilled') videoSucceeded++
                else videoFailed++
              }
            }
            steps.push({ step: 'video', result: { videoSucceeded, videoFailed, total: videoSlots.length } })
          } else {
            steps.push({ step: 'video', result: { message: 'All slots already have video' } })
          }
        }

        // Get final coverage
        const finalCoverage = await getDayCoverage(sql, day)

        return NextResponse.json({
          success: true, action: 'full-day', day,
          steps, finalCoverage,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      default:
        return NextResponse.json(
          { success: false, error: `Unknown action: ${action}` },
          { status: 400 }
        )
    }
  } catch (error) {
    console.error('[pipeline/full-coverage] POST error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
