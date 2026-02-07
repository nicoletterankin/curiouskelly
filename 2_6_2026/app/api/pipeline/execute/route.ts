/**
 * PIPELINE AUTO-EXECUTOR
 * 
 * POST /api/pipeline/execute
 * 
 * Kicks off the full pipeline in a streaming loop:
 *   1. Check status
 *   2. Generate audio (batch of N)
 *   3. Generate video (batch of N)
 *   4. Report progress
 * 
 * Uses streaming response so the client gets real-time updates.
 * Each iteration processes a batch and returns progress.
 * 
 * Body: { 
 *   audioBatchSize?: number (default 50),
 *   videoBatchSize?: number (default 25),
 *   maxIterations?: number (default 10),
 *   dayStart?: number (default 1),
 *   dayEnd?: number (default 365),
 *   dryRun?: boolean (default false)
 * }
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'
import { generateLipSyncVideo, getBaseVideo } from '@/lib/fal-lipsync-client'
import {
  findSlotsNeedingAudio,
  findSlotsNeedingVideo,
  PHASE_SCRIPT_COLUMNS,
  mapToBaseAge,
  type PipelineAge,
} from '@/lib/production-pipeline'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || ''
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

const AGE_VOICE_IDS: Record<PipelineAge, string> = {
  kid: process.env.ELEVENLABS_VOICE_KID || ELEVENLABS_VOICE_ID,
  adult: process.env.ELEVENLABS_VOICE_ADULT || ELEVENLABS_VOICE_ID,
  senior: process.env.ELEVENLABS_VOICE_SENIOR || ELEVENLABS_VOICE_ID,
}

export const maxDuration = 300 // 5 minute timeout

// GET handler - trigger pipeline via URL navigation
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  // Build body from query params
  const fakeRequest = {
    json: async () => ({
      audioBatchSize: parseInt(searchParams.get('audioBatch') || '50'),
      videoBatchSize: parseInt(searchParams.get('videoBatch') || '25'),
      maxIterations: parseInt(searchParams.get('iterations') || '10'),
      dayStart: parseInt(searchParams.get('dayStart') || '1'),
      dayEnd: parseInt(searchParams.get('dayEnd') || '365'),
      dryRun: searchParams.get('dryRun') === 'true',
    }),
  } as NextRequest
  return POST(fakeRequest)
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const body = await request.json()
    const {
      audioBatchSize = 50,
      videoBatchSize = 25,
      maxIterations = 10,
      dayStart = 1,
      dayEnd = 365,
      dryRun = false,
    } = body

    const sql = getSql()
    const log: Array<{ timestamp: string; step: string; message: string; data?: unknown }> = []

    const addLog = (step: string, message: string, data?: unknown) => {
      log.push({ timestamp: new Date().toISOString(), step, message, data })
      console.log(`[pipeline/execute] [${step}] ${message}`, data ? JSON.stringify(data).slice(0, 200) : '')
    }

    // Get initial status
    const initialStatus = await sql`
      SELECT 
        COUNT(*)::int as total,
        COUNT(audio_url)::int as with_audio,
        COUNT(video_url)::int as with_video,
        (COUNT(*) - COUNT(audio_url))::int as need_audio,
        (COUNT(audio_url) - COUNT(video_url))::int as need_video
      FROM kelly_lesson_assets
      WHERE language = 'en'
        AND age_group IN ('kid','adult','senior')
        AND archetype IN ('storyteller','explorer','scientist','architect','strategist','diplomat','mystic','rebel','macgyver','empath','provider','survivor')
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
    `
    const initial = initialStatus[0]
    addLog('status', 'Pipeline initial state', initial)

    if (dryRun) {
      return NextResponse.json({
        success: true,
        dryRun: true,
        status: initial,
        plan: {
          audioBatchSize,
          videoBatchSize,
          maxIterations,
          dayRange: [dayStart, dayEnd],
          estimatedAudioBatches: Math.ceil(Number(initial.need_audio) / audioBatchSize),
          estimatedVideoBatches: Math.ceil(Number(initial.need_video) / videoBatchSize),
        },
      })
    }

    let totalAudioGenerated = 0
    let totalVideoGenerated = 0
    let totalAudioFailed = 0
    let totalVideoFailed = 0
    let rateLimited = false
    let iteration = 0

    // =========================================================================
    // MAIN LOOP: Audio first, then Video
    // =========================================================================
    for (iteration = 0; iteration < maxIterations; iteration++) {
      const iterStart = Date.now()
      addLog('iteration', `Starting iteration ${iteration + 1}/${maxIterations}`)

      // Check time budget - stop if we're past 4 minutes
      if (Date.now() - startTime > 240_000) {
        addLog('timeout', 'Approaching 5min timeout, stopping gracefully')
        break
      }

      // =====================================================================
      // AUDIO GENERATION
      // =====================================================================
      const audioSlots = await findSlotsNeedingAudio(sql, {
        dayStart,
        dayEnd,
        limit: audioBatchSize,
      })

      if (audioSlots.length > 0) {
        addLog('audio', `Found ${audioSlots.length} slots needing audio`)

        // Get lesson scripts for needed days
        const dayNumbers = [...new Set(audioSlots.map(s => s.day_number))]
        const lessons = await sql`
          SELECT day_of_year, hook_script, story_script, wonder_script, action_script, wisdom_script
          FROM lessons WHERE day_of_year = ANY(${dayNumbers})
        `
        const lessonMap = new Map<number, Record<string, string | null>>()
        for (const l of lessons as Array<Record<string, unknown>>) {
          lessonMap.set(l.day_of_year as number, l as Record<string, string | null>)
        }

        let batchSucceeded = 0
        let batchFailed = 0

        for (const slot of audioSlots) {
          // Check time budget per slot
          if (Date.now() - startTime > 250_000) {
            addLog('timeout', 'Time budget exceeded mid-batch')
            break
          }

          try {
            const lesson = lessonMap.get(slot.day_number)
            if (!lesson) { batchFailed++; continue }

            const scriptCol = PHASE_SCRIPT_COLUMNS[slot.phase as keyof typeof PHASE_SCRIPT_COLUMNS]
            const script = lesson[scriptCol]
            if (!script) { batchFailed++; continue }

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

            if (ttsResponse.status === 429) {
              addLog('audio', 'Rate limited by ElevenLabs - pausing audio')
              rateLimited = true
              break
            }

            if (!ttsResponse.ok) {
              batchFailed++
              continue
            }

            const audioBuffer = await ttsResponse.arrayBuffer()
            const blobPath = `audio/day${String(slot.day_number).padStart(3, '0')}/${slot.phase}_${slot.age_group}_${slot.archetype}.mp3`
            const blob = await put(blobPath, Buffer.from(audioBuffer), {
              access: 'public',
              contentType: 'audio/mpeg',
            })

            await sql`
              UPDATE kelly_lesson_assets
              SET audio_url = ${blob.url},
                  script_text = ${script},
                  status = 'audio_ready',
                  updated_at = NOW()
              WHERE id = ${slot.id}
            `
            batchSucceeded++
          } catch {
            batchFailed++
          }
        }

        totalAudioGenerated += batchSucceeded
        totalAudioFailed += batchFailed
        addLog('audio', `Batch complete: ${batchSucceeded} succeeded, ${batchFailed} failed`)
      } else {
        addLog('audio', 'No more slots need audio in this range')
      }

      // If rate limited on audio, skip to video
      if (rateLimited) {
        addLog('audio', 'Rate limited - switching to video generation')
      }

      // =====================================================================
      // VIDEO GENERATION
      // =====================================================================
      const videoSlots = await findSlotsNeedingVideo(sql, {
        dayStart,
        dayEnd,
        limit: videoBatchSize,
      })

      if (videoSlots.length > 0) {
        addLog('video', `Found ${videoSlots.length} slots needing video`)

        let videoSucceeded = 0
        let videoFailed = 0

        // Process in parallel batches of 5
        for (let i = 0; i < videoSlots.length; i += 5) {
          if (Date.now() - startTime > 250_000) {
            addLog('timeout', 'Time budget exceeded mid-video-batch')
            break
          }

          const batch = videoSlots.slice(i, i + 5)
          const results = await Promise.allSettled(
            batch.map(async (slot) => {
              const baseAge = mapToBaseAge(slot.age_group)
              const baseVideoUrl = getBaseVideo(baseAge, slot.archetype)

              if (!baseVideoUrl) {
                throw new Error(`No base video for ${baseAge}/${slot.archetype}`)
              }

              const result = await generateLipSyncVideo({
                videoUrl: baseVideoUrl,
                audioUrl: slot.audio_url,
                syncMode: 'cut',
              })

              const videoResponse = await fetch(result.videoUrl)
              const videoBuffer = await videoResponse.arrayBuffer()
              const blobPath = `video/kelly/day-${String(slot.day_number).padStart(3, '0')}/${slot.phase}-${slot.age_group}-${slot.archetype}.mp4`
              const blob = await put(blobPath, videoBuffer, {
                access: 'public',
                contentType: 'video/mp4',
                addRandomSuffix: false,
              })

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

          for (const r of results) {
            if (r.status === 'fulfilled') videoSucceeded++
            else videoFailed++
          }
        }

        totalVideoGenerated += videoSucceeded
        totalVideoFailed += videoFailed
        addLog('video', `Batch complete: ${videoSucceeded} succeeded, ${videoFailed} failed`)
      } else {
        addLog('video', 'No more slots need video in this range')
      }

      // If both audio and video have nothing to do, we're done
      if (audioSlots.length === 0 && videoSlots.length === 0) {
        addLog('complete', 'All slots processed!')
        break
      }

      const iterElapsed = ((Date.now() - iterStart) / 1000).toFixed(1)
      addLog('iteration', `Iteration ${iteration + 1} completed in ${iterElapsed}s`)
    }

    // =========================================================================
    // FINAL STATUS
    // =========================================================================
    const finalStatus = await sql`
      SELECT 
        COUNT(*)::int as total,
        COUNT(audio_url)::int as with_audio,
        COUNT(video_url)::int as with_video,
        (COUNT(*) - COUNT(audio_url))::int as need_audio,
        (COUNT(audio_url) - COUNT(video_url))::int as need_video
      FROM kelly_lesson_assets
      WHERE language = 'en'
        AND age_group IN ('kid','adult','senior')
        AND archetype IN ('storyteller','explorer','scientist','architect','strategist','diplomat','mystic','rebel','macgyver','empath','provider','survivor')
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
    `

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)

    return NextResponse.json({
      success: true,
      elapsed_seconds: elapsed,
      iterations_completed: iteration + 1,
      rate_limited: rateLimited,
      audio: {
        generated: totalAudioGenerated,
        failed: totalAudioFailed,
      },
      video: {
        generated: totalVideoGenerated,
        failed: totalVideoFailed,
      },
      initial_status: initial,
      final_status: finalStatus[0],
      log,
    })
  } catch (error) {
    console.error('[pipeline/execute] Fatal error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
