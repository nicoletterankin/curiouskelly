import { NextRequest, NextResponse } from 'next/server'
import { put } from '@vercel/blob'
import { heygenClient, getAvatarConfig, type AgeCategory, type Archetype } from '@/lib/heygen-client'
import { sql } from '@/lib/db' // Declare the sql variable

// Map database age_group values to our AgeCategory
function normalizeAge(age: string): AgeCategory {
  if (age === 'kid' || age === 'child' || age === 'toddler') return 'kid'
  if (age === 'elder' || age === 'senior') return 'senior'
  return 'adult'
}

// Map database archetype values to our Archetype (handle aliases)
function normalizeArchetype(arch: string): Archetype {
  const aliases: Record<string, Archetype> = {
    coach: 'provider',
    philosopher: 'mystic',
    guardian: 'empath',
    inventor: 'macgyver',
  }
  return (aliases[arch] || arch) as Archetype
}

/**
 * POST /api/heygen/pipeline
 * 
 * Full pipeline: Generate video, poll for completion, upload to Blob, update DB
 * 
 * Body: { day_number, phase, age_group, archetype }
 */
export async function POST(request: NextRequest) {
  const startTime = Date.now()
  const logs: string[] = []
  const log = (msg: string) => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
    logs.push(`[${elapsed}s] ${msg}`)
    console.log(`[v0 Pipeline] ${msg}`)
  }

  try {
    const body = await request.json()
    const { day_number, phase, age_group, archetype } = body

    if (!day_number || !phase || !age_group || !archetype) {
      return NextResponse.json({ 
        error: 'Missing required fields', 
        required: ['day_number', 'phase', 'age_group', 'archetype'] 
      }, { status: 400 })
    }

    log(`Starting pipeline for Day ${day_number} ${phase} ${age_group}/${archetype}`)

    // STEP 1: Get audio_url from database
    log('Step 1: Fetching audio from database...')
    const assets = await sql`
      SELECT id, audio_url, script_text, video_url, video_id 
      FROM kelly_lesson_assets 
      WHERE day_number = ${day_number} 
        AND phase = ${phase} 
        AND age_group = ${age_group} 
        AND archetype = ${archetype}
      LIMIT 1
    `

    if (!assets[0]) {
      return NextResponse.json({ 
        error: 'No asset found for this combination',
        query: { day_number, phase, age_group, archetype }
      }, { status: 404 })
    }

    const asset = assets[0]
    
    // Check if already has working video
    if (asset.video_url?.includes('blob.vercel')) {
      log('Already has working Blob video URL, skipping generation')
      return NextResponse.json({
        success: true,
        skipped: true,
        message: 'Already has working video',
        video_url: asset.video_url,
        logs
      })
    }

    const audioUrl = asset.audio_url as string | null
    if (!audioUrl) {
      return NextResponse.json({ 
        error: 'No audio_url found - need audio first',
        asset_id: asset.id
      }, { status: 400 })
    }

    log(`Found audio: ${audioUrl.substring(0, 60)}...`)

    // STEP 2: Get avatar config
    log('Step 2: Getting avatar configuration...')
    const normalizedAge = normalizeAge(age_group)
    const normalizedArch = normalizeArchetype(archetype)
    
    let avatarConfig
    try {
      avatarConfig = getAvatarConfig(normalizedAge, normalizedArch)
    } catch (e) {
      return NextResponse.json({
        error: 'Invalid avatar config',
        age_group,
        normalized_age: normalizedAge,
        archetype,
        normalized_archetype: normalizedArch,
        details: e instanceof Error ? e.message : String(e)
      }, { status: 400 })
    }

    log(`Avatar: ${avatarConfig.avatarGroupId}, Look: ${avatarConfig.lookId}`)

    // STEP 3: Call HeyGen to generate video
    log('Step 3: Calling HeyGen API to generate video...')
    let videoResponse
    try {
      videoResponse = await heygenClient.generateVideoWithAudio(
        avatarConfig.avatarGroupId,
        avatarConfig.lookId,
        audioUrl,
        `Day ${day_number} - ${phase} - ${age_group}/${archetype}`
      )
    } catch (e) {
      log(`HeyGen API error: ${e instanceof Error ? e.message : String(e)}`)
      return NextResponse.json({
        error: 'HeyGen API failed',
        details: e instanceof Error ? e.message : String(e),
        logs
      }, { status: 500 })
    }

    const videoId = videoResponse.video_id
    log(`HeyGen video_id: ${videoId}`)

    // STEP 4: Update database with video_id (processing state)
    log('Step 4: Updating database with video_id...')
    await sql`
      UPDATE kelly_lesson_assets 
      SET video_id = ${videoId}, status = 'processing', updated_at = NOW()
      WHERE id = ${asset.id}
    `

    // Store in heygen_videos table
    await sql`
      INSERT INTO heygen_videos (
        engine, heygen_video_id, day_of_year, phase, age_category, archetype, status, created_at
      ) VALUES (
        'heygen', ${videoId}, ${day_number}, ${phase}, ${age_group}, ${archetype}, 'processing', NOW()
      )
      ON CONFLICT (heygen_video_id) DO UPDATE SET status = 'processing', updated_at = NOW()
    `
    .catch(() => {})

    // STEP 5: Poll for completion (max 5 minutes)
    log('Step 5: Polling HeyGen for completion...')
    let finalStatus
    try {
      finalStatus = await heygenClient.waitForVideo(videoId, {
        maxWaitMs: 300000, // 5 minutes
        pollIntervalMs: 5000 // Every 5 seconds
      })
    } catch (e) {
      log(`Polling error: ${e instanceof Error ? e.message : String(e)}`)
      await sql`
        UPDATE kelly_lesson_assets SET status = 'failed', updated_at = NOW() WHERE id = ${asset.id}
      `
      await sql`
        UPDATE heygen_videos SET status = 'failed', error_message = ${e instanceof Error ? e.message : 'Unknown error'} WHERE heygen_video_id = ${videoId}
      `
      .catch(() => {})
      return NextResponse.json({
        error: 'Video generation failed or timed out',
        video_id: videoId,
        details: e instanceof Error ? e.message : String(e),
        logs
      }, { status: 500 })
    }

    const heygenVideoUrl = finalStatus.video_url
    if (!heygenVideoUrl) {
      return NextResponse.json({
        error: 'No video_url in HeyGen response',
        status: finalStatus,
        logs
      }, { status: 500 })
    }

    log(`HeyGen video ready: ${heygenVideoUrl.substring(0, 60)}...`)

    // STEP 6: Download video and upload to Vercel Blob
    log('Step 6: Downloading from HeyGen and uploading to Vercel Blob...')
    let blobUrl
    try {
      const videoResponse = await fetch(heygenVideoUrl)
      if (!videoResponse.ok) {
        throw new Error(`Failed to download: ${videoResponse.status}`)
      }
      
      const videoBuffer = await videoResponse.arrayBuffer()
      const filename = `kelly/videos/day_${day_number}_${phase}_${age_group}_${archetype}.mp4`
      
      const blob = await put(filename, videoBuffer, {
        access: 'public',
        contentType: 'video/mp4'
      })
      
      blobUrl = blob.url
    } catch (e) {
      log(`Blob upload error: ${e instanceof Error ? e.message : String(e)}`)
      // Still update with HeyGen URL as fallback
      await sql`
        UPDATE kelly_lesson_assets 
        SET video_url = ${heygenVideoUrl}, status = 'completed', updated_at = NOW()
        WHERE id = ${asset.id}
      `
      return NextResponse.json({
        success: true,
        warning: 'Blob upload failed, using HeyGen URL directly',
        video_url: heygenVideoUrl,
        video_id: videoId,
        logs
      })
    }

    log(`Uploaded to Blob: ${blobUrl}`)

    // STEP 7: Update database with final Blob URL
    log('Step 7: Updating database with Blob URL...')
    await sql`
      UPDATE kelly_lesson_assets 
      SET video_url = ${blobUrl}, status = 'completed', updated_at = NOW()
      WHERE id = ${asset.id}
    `
    await sql`
      UPDATE heygen_videos 
      SET status = 'completed', video_url = ${blobUrl}, updated_at = NOW()
      WHERE heygen_video_id = ${videoId}
    `
    .catch(() => {})

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
    log(`COMPLETE in ${elapsed}s`)

    return NextResponse.json({
      success: true,
      video_url: blobUrl,
      video_id: videoId,
      heygen_url: heygenVideoUrl,
      elapsed_seconds: parseFloat(elapsed),
      details: {
        day_number,
        phase,
        age_group,
        archetype,
        normalized_age: normalizedAge,
        normalized_archetype: normalizedArch,
        avatar_id: avatarConfig.avatarGroupId,
        look_id: avatarConfig.lookId
      },
      logs
    })

  } catch (error) {
    console.error('[v0 Pipeline] Unexpected error:', error)
    return NextResponse.json({
      error: 'Pipeline failed',
      details: error instanceof Error ? error.message : String(error),
      logs
    }, { status: 500 })
  }
}

/**
 * GET /api/heygen/pipeline
 * 
 * Check what's ready to process
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '18')

  // Get all assets for this day that have audio but no working video
  const ready = await sql`
    SELECT day_number, phase, age_group, archetype, 
           audio_url IS NOT NULL as has_audio,
           video_url,
           video_id,
           status
    FROM kelly_lesson_assets 
    WHERE day_number = ${day}
    ORDER BY phase, age_group, archetype
  `

  const needsVideo = ready.filter((r: Record<string, unknown>) => 
    r.has_audio && (!r.video_url || !(r.video_url as string).includes('blob.vercel'))
  )

  const hasWorkingVideo = ready.filter((r: Record<string, unknown>) => 
    r.video_url && (r.video_url as string).includes('blob.vercel')
  )

  return NextResponse.json({
    day,
    total_assets: ready.length,
    needs_video: needsVideo.length,
    has_working_video: hasWorkingVideo.length,
    ready_to_process: needsVideo.map((r: Record<string, unknown>) => ({
      day_number: r.day_number,
      phase: r.phase,
      age_group: r.age_group,
      archetype: r.archetype,
      current_video_url: r.video_url,
      current_status: r.status
    })),
    already_complete: hasWorkingVideo.map((r: Record<string, unknown>) => ({
      day_number: r.day_number,
      phase: r.phase,
      age_group: r.age_group,
      archetype: r.archetype,
      video_url: r.video_url
    })),
    next_step: needsVideo.length > 0 
      ? `POST /api/heygen/pipeline with body: ${JSON.stringify({ day_number: day, phase: needsVideo[0].phase, age_group: needsVideo[0].age_group, archetype: needsVideo[0].archetype })}`
      : 'All videos complete!'
  })
}
