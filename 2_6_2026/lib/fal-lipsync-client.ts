// ============================================================================
// FAL.AI LIP-SYNC CLIENT
// Redubs existing Kelly videos with new audio for any lesson
// ============================================================================

import { fal } from '@fal-ai/client'

// Configure fal client
fal.config({
  credentials: process.env.FAL_KEY!,
})

export interface LipSyncRequest {
  videoUrl: string      // Base video URL (from Day 18 HeyGen output)
  audioUrl: string      // New audio URL (from ElevenLabs)
  syncMode?: 'cut' | 'loop' | 'freeze'  // How to handle length mismatch
}

export interface LipSyncResult {
  videoUrl: string      // Output video with new lip-sync
  duration: number      // Video duration in seconds
}

/**
 * REUSABLE BASE VIDEOS
 * These are HeyGen videos that can be redubbed for ANY lesson via FAL.ai lip-sync.
 * Key: {age}_{archetype}
 * Value: Blob URL of completed HeyGen video
 * 
 * GOAL: All 36 variants (3 ages x 12 archetypes) need base videos.
 * Use POST /api/heygen/kickoff-all-36 to generate missing ones.
 */
export const BASE_VIDEOS: Record<string, string> = {
  // -------------------------------------------------------------------------
  // ADULT ARCHETYPES (12)
  // -------------------------------------------------------------------------
  'adult_storyteller': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-storyteller.mp4',
  'adult_explorer': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-explorer.mp4',
  'adult_scientist': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-scientist.mp4',
  'adult_architect': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-architect.mp4',
  'adult_strategist': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-strategist.mp4',
  'adult_diplomat': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-diplomat.mp4',
  'adult_mystic': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-mystic.mp4',
  'adult_rebel': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-rebel.mp4',
  'adult_macgyver': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-macgyver.mp4',
  'adult_empath': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-empath.mp4',
  'adult_provider': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-provider.mp4',
  'adult_survivor': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-survivor.mp4',
  
  // -------------------------------------------------------------------------
  // KID ARCHETYPES (12)
  // -------------------------------------------------------------------------
  'kid_storyteller': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-storyteller.mp4',
  'kid_explorer': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-explorer.mp4',
  'kid_scientist': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-scientist.mp4',
  'kid_architect': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-architect.mp4',
  'kid_strategist': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-strategist.mp4',
  'kid_diplomat': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-diplomat.mp4',
  'kid_mystic': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-mystic.mp4',
  'kid_rebel': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-rebel.mp4',
  'kid_macgyver': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-macgyver.mp4',
  'kid_empath': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-empath.mp4',
  'kid_provider': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-provider.mp4',
  'kid_survivor': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/kid-survivor.mp4',
  
  // -------------------------------------------------------------------------
  // SENIOR ARCHETYPES (12)
  // -------------------------------------------------------------------------
  'senior_storyteller': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-storyteller.mp4',
  'senior_explorer': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-explorer.mp4',
  'senior_scientist': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-scientist.mp4',
  'senior_architect': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-architect.mp4',
  'senior_strategist': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-strategist.mp4',
  'senior_diplomat': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-diplomat.mp4',
  'senior_mystic': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-mystic.mp4',
  'senior_rebel': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-rebel.mp4',
  'senior_macgyver': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-macgyver.mp4',
  'senior_empath': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-empath.mp4',
  'senior_provider': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-provider.mp4',
  'senior_survivor': 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/senior-survivor.mp4',
}

/**
 * Get base video URL for a given age and archetype
 * Falls back to adult_storyteller if specific combo not available
 */
export function getBaseVideo(age: 'kid' | 'adult' | 'senior', archetype: string): string {
  const key = `${age}_${archetype}`
  
  // Try exact match first
  if (BASE_VIDEOS[key]) {
    return BASE_VIDEOS[key]
  }
  
  // Fallback: any video for that age
  const ageVideos = Object.entries(BASE_VIDEOS).filter(([k]) => k.startsWith(age))
  if (ageVideos.length > 0) {
    return ageVideos[0][1]
  }
  
  // Ultimate fallback: adult storyteller
  return BASE_VIDEOS['adult_storyteller']
}

/**
 * Generate a lip-synced video using fal.ai
 * Takes an existing Kelly video and redubs it with new audio
 */
export async function generateLipSyncVideo(request: LipSyncRequest): Promise<LipSyncResult> {
  console.log('[fal.ai] Starting lip-sync generation...')
  console.log('[fal.ai] Video:', request.videoUrl)
  console.log('[fal.ai] Audio:', request.audioUrl)
  
  try {
    // Use fal.ai's sync-lipsync model
    // Docs: https://fal.ai/models/fal-ai/sync-lipsync
    const result = await fal.subscribe('fal-ai/sync-lipsync', {
      input: {
        video_url: request.videoUrl,
        audio_url: request.audioUrl,
        sync_mode: request.syncMode || 'cut',  // Cut audio to match video length
      },
      logs: true,
      onQueueUpdate: (update) => {
        if (update.status === 'IN_PROGRESS') {
          console.log('[fal.ai] Processing...', update.logs?.map(l => l.message).join(' '))
        }
      },
    })
    
    console.log('[fal.ai] Complete!')
    
    return {
      videoUrl: result.data.video.url,
      duration: result.data.video.duration || 0,
    }
  } catch (error) {
    console.error('[fal.ai] Error:', error)
    throw error
  }
}

/**
 * High-level function: Generate video for any day/phase/age/archetype
 * Combines base video selection + lip-sync in one call
 */
export async function generateLessonVideo(params: {
  audioUrl: string
  age: 'kid' | 'adult' | 'senior'
  archetype: string
}): Promise<LipSyncResult> {
  const baseVideo = getBaseVideo(params.age, params.archetype)
  
  console.log(`[LessonVideo] Age: ${params.age}, Archetype: ${params.archetype}`)
  console.log(`[LessonVideo] Base video: ${baseVideo}`)
  
  return generateLipSyncVideo({
    videoUrl: baseVideo,
    audioUrl: params.audioUrl,
    syncMode: 'cut',
  })
}

/**
 * Check how many base videos we have coverage for
 */
export function getBaseVideoCoverage(): {
  total: number
  covered: number
  missing: string[]
} {
  const allCombos = [
    'kid', 'adult', 'senior'
  ].flatMap(age => [
    'storyteller', 'explorer', 'scientist', 'architect', 'strategist',
    'diplomat', 'mystic', 'rebel', 'macgyver', 'empath', 'provider', 'survivor'
  ].map(arch => `${age}_${arch}`))
  
  const covered = allCombos.filter(combo => BASE_VIDEOS[combo])
  const missing = allCombos.filter(combo => !BASE_VIDEOS[combo])
  
  return {
    total: allCombos.length,
    covered: covered.length,
    missing,
  }
}
