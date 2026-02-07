// ============================================================================
// VIDEO ORCHESTRATOR
// Coordinates ElevenLabs audio + HeyGen video generation + storage
// ============================================================================

import { generateKellyVideo, heygenClient, type Archetype, type AgeCategory } from './heygen-client'
import { generateScript, type Phase, type LessonData } from './script-generator'
import { put } from '@vercel/blob'

// ============================================================================
// TYPES
// ============================================================================

export interface VideoGenerationRequest {
  dayOfYear: number
  phase: Phase
  ageCategory: AgeCategory
  archetype: Archetype
  lessonData: LessonData
  options?: {
    generateAudio?: boolean
    useExistingAudio?: string // URL to existing audio
    storeVideo?: boolean
    testMode?: boolean
  }
}

export interface VideoGenerationResult {
  success: boolean
  videoId?: string
  videoUrl?: string
  audioUrl?: string
  thumbnailUrl?: string
  duration?: number
  script?: string
  error?: string
  storedPath?: string
}

export interface BatchGenerationResult {
  dayOfYear: number
  ageCategory: AgeCategory
  archetype: Archetype
  phases: Record<Phase, VideoGenerationResult>
  totalDuration: number
  success: boolean
}

// ============================================================================
// CONSTANTS
// ============================================================================

const PHASES: Phase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']

// ============================================================================
// SINGLE VIDEO GENERATION
// ============================================================================

/**
 * Generate a single video for a specific lesson phase
 */
export async function generateLessonPhaseVideo(
  request: VideoGenerationRequest
): Promise<VideoGenerationResult> {
  const { dayOfYear, phase, ageCategory, archetype, lessonData, options = {} } = request
  const { generateAudio = false, useExistingAudio, storeVideo = true, testMode = false } = options

  try {
    // Step 1: Generate script
    const scriptResult = generateScript(lessonData, phase, archetype, ageCategory)
    const script = scriptResult.script

    console.log(`[VideoOrchestrator] Day ${dayOfYear} ${phase} - Script generated (${scriptResult.estimatedDuration}s)`)

    // Step 2: Get or generate audio
    let audioUrl = useExistingAudio

    if (generateAudio && !audioUrl) {
      // Generate audio via ElevenLabs (implement separately)
      // For now, we'll use HeyGen's TTS
      audioUrl = undefined
    }

    // Step 3: Generate video via HeyGen
    const videoTitle = `Day ${dayOfYear} - ${phase.toUpperCase()} - ${ageCategory}_${archetype}`
    
    const heygenResult = await generateKellyVideo({
      age: ageCategory,
      archetype,
      script: audioUrl ? '' : script,
      audioUrl,
      title: videoTitle,
      test: testMode,
    })

    console.log(`[VideoOrchestrator] Day ${dayOfYear} ${phase} - HeyGen video started: ${heygenResult.video_id}`)

    // Step 4: Wait for video completion
    const videoStatus = await heygenClient.waitForVideo(heygenResult.video_id, {
      maxWaitMs: 5 * 60 * 1000, // 5 minutes max
      pollIntervalMs: 5000,
    })

    if (videoStatus.status !== 'completed' || !videoStatus.video_url) {
      throw new Error(`Video generation failed: ${videoStatus.error || 'Unknown error'}`)
    }

    console.log(`[VideoOrchestrator] Day ${dayOfYear} ${phase} - Video completed: ${videoStatus.video_url}`)

    // Step 5: Store video in Vercel Blob (optional)
    let storedPath: string | undefined

    if (storeVideo && videoStatus.video_url) {
      const blobPath = `videos/day-${String(dayOfYear).padStart(3, '0')}/${ageCategory}/${archetype}/${phase}.mp4`
      
      try {
        // Download and re-upload to our blob storage
        const videoResponse = await fetch(videoStatus.video_url)
        const videoBlob = await videoResponse.blob()
        
        const { url } = await put(blobPath, videoBlob, {
          access: 'public',
          contentType: 'video/mp4',
        })
        
        storedPath = url
        console.log(`[VideoOrchestrator] Day ${dayOfYear} ${phase} - Stored at: ${storedPath}`)
      } catch (storeError) {
        console.warn(`[VideoOrchestrator] Failed to store video:`, storeError)
        // Continue without storing - use HeyGen URL directly
      }
    }

    return {
      success: true,
      videoId: heygenResult.video_id,
      videoUrl: storedPath || videoStatus.video_url,
      thumbnailUrl: videoStatus.thumbnail_url,
      duration: videoStatus.duration,
      script,
      storedPath,
    }
  } catch (error) {
    console.error(`[VideoOrchestrator] Day ${dayOfYear} ${phase} - Error:`, error)
    
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }
  }
}

// ============================================================================
// BATCH VIDEO GENERATION
// ============================================================================

/**
 * Generate all 5 phase videos for a lesson
 */
export async function generateFullLessonVideos(
  dayOfYear: number,
  ageCategory: AgeCategory,
  archetype: Archetype,
  lessonData: LessonData,
  options?: {
    phases?: Phase[]
    testMode?: boolean
    storeVideos?: boolean
    sequential?: boolean // If true, generate one at a time
  }
): Promise<BatchGenerationResult> {
  const {
    phases = PHASES,
    testMode = false,
    storeVideos = true,
    sequential = true,
  } = options || {}

  console.log(`[VideoOrchestrator] Starting batch generation for Day ${dayOfYear} (${ageCategory}_${archetype})`)

  const results: Record<Phase, VideoGenerationResult> = {} as Record<Phase, VideoGenerationResult>
  let totalDuration = 0
  let hasErrors = false

  if (sequential) {
    // Generate one at a time (safer, respects rate limits)
    for (const phase of phases) {
      const result = await generateLessonPhaseVideo({
        dayOfYear,
        phase,
        ageCategory,
        archetype,
        lessonData,
        options: { testMode, storeVideo: storeVideos },
      })

      results[phase] = result
      
      if (result.success && result.duration) {
        totalDuration += result.duration
      } else {
        hasErrors = true
      }

      // Small delay between phases
      await new Promise(r => setTimeout(r, 1000))
    }
  } else {
    // Generate in parallel (faster but may hit rate limits)
    const promises = phases.map(phase =>
      generateLessonPhaseVideo({
        dayOfYear,
        phase,
        ageCategory,
        archetype,
        lessonData,
        options: { testMode, storeVideo: storeVideos },
      }).then(result => ({ phase, result }))
    )

    const settled = await Promise.allSettled(promises)

    for (const outcome of settled) {
      if (outcome.status === 'fulfilled') {
        const { phase, result } = outcome.value
        results[phase] = result
        
        if (result.success && result.duration) {
          totalDuration += result.duration
        } else {
          hasErrors = true
        }
      } else {
        hasErrors = true
      }
    }
  }

  console.log(`[VideoOrchestrator] Batch complete for Day ${dayOfYear}: ${hasErrors ? 'with errors' : 'success'}`)

  return {
    dayOfYear,
    ageCategory,
    archetype,
    phases: results,
    totalDuration,
    success: !hasErrors,
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Check if videos exist for a lesson
 */
export async function checkLessonVideosExist(
  dayOfYear: number,
  ageCategory: AgeCategory,
  archetype: Archetype
): Promise<Record<Phase, boolean>> {
  const result: Record<Phase, boolean> = {} as Record<Phase, boolean>
  
  for (const phase of PHASES) {
    const blobPath = `videos/day-${String(dayOfYear).padStart(3, '0')}/${ageCategory}/${archetype}/${phase}.mp4`
    
    try {
      // Try to HEAD the blob to check existence
      const response = await fetch(`https://${process.env.BLOB_READ_WRITE_TOKEN?.split('_')[0]}.public.blob.vercel-storage.com/${blobPath}`, {
        method: 'HEAD',
      })
      result[phase] = response.ok
    } catch {
      result[phase] = false
    }
  }
  
  return result
}

/**
 * Get video URL for a lesson phase
 */
export function getLessonVideoUrl(
  dayOfYear: number,
  phase: Phase,
  ageCategory: AgeCategory,
  archetype: Archetype
): string {
  const dayStr = String(dayOfYear).padStart(3, '0')
  return `/api/video/url?day=${dayStr}&phase=${phase}&age=${ageCategory}&archetype=${archetype}`
}

/**
 * Estimate credits needed for a full lesson
 */
export function estimateCreditsForLesson(phases: Phase[] = PHASES): number {
  // Rough estimate: 1 credit per 10 seconds of video
  // Average lesson phase is ~45 seconds = ~5 credits total
  return phases.length
}

/**
 * Estimate total credits for all content
 * 365 days × 5 phases × 36 avatars = 65,700 videos
 */
export function estimateTotalCredits(): {
  allContent: number
  perDay: number
  perAvatarPerDay: number
} {
  const phases = 5
  const avatars = 36
  const days = 365
  
  return {
    allContent: days * phases * avatars, // 65,700
    perDay: phases * avatars, // 180
    perAvatarPerDay: phases, // 5
  }
}
