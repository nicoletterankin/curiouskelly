/**
 * AUDIO SEEDER FOR kelly_lesson_assets
 * 
 * Bridges the gap between:
 *   - lesson_audio table (where /api/audio/generate writes)
 *   - kelly_lesson_assets table (where /api/fal/auto-run reads from)
 * 
 * Also handles directly generating audio for slots that have no audio anywhere.
 * 
 * GET /api/pipeline/seed-audio?dayStart=1&dayEnd=30
 *   Returns: which slots need audio, which have audio in lesson_audio but not kelly_lesson_assets
 * 
 * POST /api/pipeline/seed-audio
 *   Body: { action: "bridge", dayStart: 1, dayEnd: 30 }
 *     -> Copies audio_urls from lesson_audio to matching kelly_lesson_assets rows
 *   
 *   Body: { action: "generate", dayStart: 1, dayEnd: 30, limit: 50 }
 *     -> Generates new ElevenLabs audio directly into kelly_lesson_assets
 */

import { NextRequest, NextResponse } from 'next/server'
import { getSql } from '@/lib/db'
import { put } from '@vercel/blob'
import {
  seedDayRange,
  findSlotsNeedingAudio,
  PHASE_SCRIPT_COLUMNS,
  type PipelineAge,
} from '@/lib/production-pipeline'

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || ''
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '1bd001e7e50f421d891986aad5158bc8'

// ============================================================================
// GET - Diagnostic: show what needs audio
// ============================================================================

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const dayStart = parseInt(searchParams.get('dayStart') || '1')
  const dayEnd = parseInt(searchParams.get('dayEnd') || '30')

  try {
    const sql = getSql()

    // Count slots in kelly_lesson_assets without audio
    const needsAudio = await sql`
      SELECT day_number, COUNT(*) as count
      FROM kelly_lesson_assets
      WHERE audio_url IS NULL
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
        AND language = 'en'
      GROUP BY day_number
      ORDER BY day_number
    `

    // Count audio in lesson_audio table for this range
    const existingInLessonAudio = await sql`
      SELECT day_number, COUNT(*) as count
      FROM lesson_audio
      WHERE day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
        AND audio_url IS NOT NULL
      GROUP BY day_number
      ORDER BY day_number
    `

    // Count already-bridged rows in kelly_lesson_assets
    const alreadyHasAudio = await sql`
      SELECT day_number, COUNT(*) as count
      FROM kelly_lesson_assets
      WHERE audio_url IS NOT NULL
        AND day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
        AND language = 'en'
      GROUP BY day_number
      ORDER BY day_number
    `

    // Count total seeded slots
    const totalSeeded = await sql`
      SELECT COUNT(*) as count
      FROM kelly_lesson_assets
      WHERE day_number >= ${dayStart}
        AND day_number <= ${dayEnd}
        AND language = 'en'
    `

    return NextResponse.json({
      success: true,
      range: { dayStart, dayEnd },
      totalSeededSlots: parseInt((totalSeeded[0]?.count as string) || '0'),
      slotsNeedingAudioByDay: needsAudio,
      audioInLessonAudioByDay: existingInLessonAudio,
      slotsWithAudioByDay: alreadyHasAudio,
      hint: 'POST with action="bridge" to copy from lesson_audio, or action="generate" to create new audio',
    })
  } catch (error) {
    console.error('[seed-audio] GET error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

// ============================================================================
// POST - Actions: bridge or generate
// ============================================================================

interface SeedAudioRequest {
  action: 'bridge' | 'generate' | 'seed-and-bridge'
  dayStart?: number
  dayEnd?: number
  limit?: number
  dryRun?: boolean
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const body = await request.json() as SeedAudioRequest
    const { action, dayStart = 1, dayEnd = 30, limit = 100, dryRun = false } = body

    const sql = getSql()

    switch (action) {
      // ====================================================================
      // BRIDGE: Copy audio from lesson_audio to kelly_lesson_assets
      // ====================================================================
      case 'bridge': {
        // Find kelly_lesson_assets rows without audio that have matching lesson_audio
        // The lesson_audio table uses age_variant format like "adult_hook" or "kid_hook_en_explorer"
        const slotsNeedingAudio = await sql`
          SELECT k.id, k.day_number, k.phase, k.age_group, k.archetype
          FROM kelly_lesson_assets k
          WHERE k.audio_url IS NULL
            AND k.day_number >= ${dayStart}
            AND k.day_number <= ${dayEnd}
            AND k.language = 'en'
          ORDER BY k.day_number, k.phase
          LIMIT ${limit}
        `

        if (dryRun) {
          return NextResponse.json({
            success: true, action: 'bridge', dryRun: true,
            slotsToProcess: slotsNeedingAudio.length,
            sample: (slotsNeedingAudio as Array<Record<string, unknown>>).slice(0, 10),
          })
        }

        let bridged = 0
        let notFound = 0

        for (const slot of slotsNeedingAudio as Array<{ id: number; day_number: number; phase: string; age_group: string; archetype: string }>) {
          // Try multiple age_variant patterns that lesson_audio might use
          const patterns = [
            `${slot.age_group}_${slot.phase}`,
            `${slot.age_group}_${slot.phase}_en_${slot.archetype}`,
            `${slot.age_group}_${slot.phase}_en`,
          ]

          let audioUrl: string | null = null

          for (const pattern of patterns) {
            const match = await sql`
              SELECT audio_url FROM lesson_audio
              WHERE day_number = ${slot.day_number}
                AND age_variant = ${pattern}
                AND audio_url IS NOT NULL
              LIMIT 1
            `
            if (match.length > 0) {
              audioUrl = match[0].audio_url as string
              break
            }
          }

          // Also try a more flexible match - any audio for this day+phase
          if (!audioUrl) {
            const flexMatch = await sql`
              SELECT audio_url FROM lesson_audio
              WHERE day_number = ${slot.day_number}
                AND age_variant LIKE ${`%${slot.phase}%`}
                AND audio_url IS NOT NULL
              LIMIT 1
            `
            if (flexMatch.length > 0) {
              audioUrl = flexMatch[0].audio_url as string
            }
          }

          if (audioUrl) {
            await sql`
              UPDATE kelly_lesson_assets
              SET audio_url = ${audioUrl}, status = 'audio_ready', updated_at = NOW()
              WHERE id = ${slot.id}
            `
            bridged++
          } else {
            notFound++
          }
        }

        return NextResponse.json({
          success: true, action: 'bridge',
          bridged, notFound,
          total: slotsNeedingAudio.length,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // GENERATE: Create fresh ElevenLabs audio directly into kelly_lesson_assets
      // ====================================================================
      case 'generate': {
        const slots = await findSlotsNeedingAudio(sql, { dayStart, dayEnd, limit })

        if (slots.length === 0) {
          return NextResponse.json({
            success: true, action: 'generate',
            message: 'No slots need audio',
            generated: 0,
          })
        }

        if (dryRun) {
          return NextResponse.json({
            success: true, action: 'generate', dryRun: true,
            slotsNeedingAudio: slots.length,
            sample: slots.slice(0, 10),
          })
        }

        // Get lesson scripts
        const dayNumbers = [...new Set(slots.map(s => s.day_number))]
        const lessons = await sql`
          SELECT day_of_year, hook_script, story_script, wonder_script, action_script, wisdom_script
          FROM lessons WHERE day_of_year = ANY(${dayNumbers})
        `
        const lessonMap = new Map<number, Record<string, string | null>>()
        for (const l of lessons as Array<Record<string, unknown>>) {
          lessonMap.set(l.day_of_year as number, l as Record<string, string | null>)
        }

        let generated = 0
        let failed = 0
        let skipped = 0

        for (const slot of slots) {
          const lesson = lessonMap.get(slot.day_number)
          if (!lesson) { skipped++; continue }

          const scriptCol = PHASE_SCRIPT_COLUMNS[slot.phase as keyof typeof PHASE_SCRIPT_COLUMNS]
          const script = lesson[scriptCol]
          if (!script) { skipped++; continue }

          try {
            const ttsResponse = await fetch(
              `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
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
              return NextResponse.json({
                success: true, action: 'generate',
                rateLimited: true, generated, failed, skipped,
                message: 'Rate limited by ElevenLabs',
                elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
              })
            }

            if (!ttsResponse.ok) { failed++; continue }

            const audioBuffer = await ttsResponse.arrayBuffer()
            const blobPath = `audio/day${String(slot.day_number).padStart(3, '0')}/${slot.phase}_${slot.age_group}_${slot.archetype}.mp3`
            const blob = await put(blobPath, Buffer.from(audioBuffer), {
              access: 'public', contentType: 'audio/mpeg',
            })

            await sql`
              UPDATE kelly_lesson_assets
              SET audio_url = ${blob.url}, script_text = ${script}, status = 'audio_ready', updated_at = NOW()
              WHERE id = ${slot.id}
            `
            generated++
          } catch {
            failed++
          }
        }

        return NextResponse.json({
          success: true, action: 'generate',
          generated, failed, skipped,
          total: slots.length,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      // ====================================================================
      // SEED-AND-BRIDGE: Ensure slots exist then bridge audio
      // ====================================================================
      case 'seed-and-bridge': {
        // Step 1: Seed all slots for the day range
        const seedResult = await seedDayRange(sql, dayStart, dayEnd)

        if (dryRun) {
          return NextResponse.json({
            success: true, action: 'seed-and-bridge', dryRun: true,
            seedResult,
          })
        }

        // Step 2: Bridge audio from lesson_audio
        const slotsNeedingAudio = await sql`
          SELECT k.id, k.day_number, k.phase, k.age_group, k.archetype
          FROM kelly_lesson_assets k
          WHERE k.audio_url IS NULL
            AND k.day_number >= ${dayStart}
            AND k.day_number <= ${dayEnd}
            AND k.language = 'en'
          ORDER BY k.day_number, k.phase
          LIMIT ${limit}
        `

        let bridged = 0
        for (const slot of slotsNeedingAudio as Array<{ id: number; day_number: number; phase: string; age_group: string; archetype: string }>) {
          const patterns = [
            `${slot.age_group}_${slot.phase}`,
            `${slot.age_group}_${slot.phase}_en_${slot.archetype}`,
          ]

          for (const pattern of patterns) {
            const match = await sql`
              SELECT audio_url FROM lesson_audio
              WHERE day_number = ${slot.day_number}
                AND age_variant = ${pattern}
                AND audio_url IS NOT NULL
              LIMIT 1
            `
            if (match.length > 0) {
              await sql`
                UPDATE kelly_lesson_assets
                SET audio_url = ${match[0].audio_url}, status = 'audio_ready', updated_at = NOW()
                WHERE id = ${slot.id}
              `
              bridged++
              break
            }
          }
        }

        return NextResponse.json({
          success: true, action: 'seed-and-bridge',
          seedResult, bridged,
          slotsChecked: slotsNeedingAudio.length,
          elapsed: ((Date.now() - startTime) / 1000).toFixed(1),
        })
      }

      default:
        return NextResponse.json({ success: false, error: `Unknown action: ${action}` }, { status: 400 })
    }
  } catch (error) {
    console.error('[seed-audio] POST error:', error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
