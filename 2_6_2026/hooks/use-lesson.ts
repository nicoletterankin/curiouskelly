'use client'

import useSWR from 'swr'
import type { Lesson, LessonPhase, KellyArchetype, KellyLessonAsset } from '@/lib/types'

interface LessonResponse {
  lesson: Lesson
  perspective: {
    title: string
    theme: string
    topic: string
    hook_script?: string
    story_script?: string
    wonder_script?: string
    action_script?: string
    wisdom_script?: string
    phases: Record<LessonPhase, string>
  } | null
  // Kelly's word-by-word scripts per phase
  // Precedence: kelly_scripts (variant) > kelly_lesson_assets > perspective > base
  kellyScripts: Record<string, string> | null
  hasKellyScripts: boolean
  kellyVariantSource: string | null // 'kid' | 'adult' | 'elder' when served from kelly_scripts table
  displayDate: string
  dayOfYear: number
}

interface VideoUrlResponse {
  url: string | null
  audioUrl: string | null
  fallbackUrl: string | null
  status: 'ready' | 'generating' | 'error'
  thumbnailUrl: string | null
}

const fetcher = async (url: string) => {
  try {
    const res = await fetch(url, {
      cache: 'no-store',
      headers: { 'Cache-Control': 'no-cache' }
    })
    
    if (res.status === 429) return null
    if (!res.ok) {
      const text = await res.text()
      try { return JSON.parse(text) } catch { return null }
    }
    
    const text = await res.text()
    if (!text) return null
    try { return JSON.parse(text) } catch { return null }
  } catch {
    return null
  }
}

export function useLesson(
  dayOfYear: number,
  timezone: string,
  options?: {
    age?: number
    archetype?: KellyArchetype
    language?: string
  }
) {
  const params = new URLSearchParams({
    day: dayOfYear.toString(),
    timezone,
  })
  
  if (options?.age) params.set('age', options.age.toString())
  if (options?.archetype) params.set('archetype', options.archetype)
  if (options?.language) params.set('language', options.language)

  // SWR key uses all params for proper cache separation
  // Language/age/archetype changes create distinct cache keys - no timestamp needed
  // This fixes the race condition where language changes within same minute bucket returned stale data
  const swrKey = `/api/lessons/by-day?${params.toString()}&_v=6`
  
  const { data, error, isLoading, mutate } = useSWR<LessonResponse>(
    swrKey,
    fetcher,
    {
      revalidateOnFocus: false, // Disable - causes flicker
      revalidateOnMount: true, // Always fetch fresh on mount
      dedupingInterval: 2000, // 2s dedup - prevents duplicate requests while allowing language changes
      keepPreviousData: false, // Don't show stale data during language switch
      refreshInterval: 0, // No auto refresh
      revalidateIfStale: false, // Don't auto-refetch - user controls refresh
    }
  )
  
  return {
    lesson: data?.lesson || null,
    perspective: data?.perspective || null,
    kellyScripts: data?.kellyScripts || null,
    hasKellyScripts: data?.hasKellyScripts || false,
    kellyVariantSource: data?.kellyVariantSource || null,
    displayDate: data?.displayDate || '',
    isLoading,
    error,
    refresh: mutate,
  }
}

/**
 * Video Source Engines for A/B testing lip-sync quality
 * Enterprise-grade lip-sync comparison across industry-leading providers
 */
export type VideoSource = 'auto' | 'heygen' | 'fal' | 'sync' | 'musetalk'

export const VIDEO_SOURCES: { 
  id: VideoSource
  label: string
  shortLabel: string
  description: string
  color: string
  tier: 'enterprise' | 'pro' | 'open-source'
}[] = [
  { 
    id: 'heygen', 
    label: 'HeyGen', 
    shortLabel: 'HeyGen',
    description: 'Enterprise Avatar API', 
    color: '#0ea5e9',
    tier: 'enterprise'
  },
  { 
    id: 'fal', 
    label: 'Fal.ai', 
    shortLabel: 'Fal',
    description: 'Serverless GPU Cloud', 
    color: '#f97316',
    tier: 'pro'
  },
  { 
    id: 'sync', 
    label: 'Sync Labs', 
    shortLabel: 'Sync',
    description: 'Lipsync-2.0 Engine', 
    color: '#22c55e',
    tier: 'pro'
  },
  { 
    id: 'musetalk', 
    label: 'MuseTalk', 
    shortLabel: 'Muse',
    description: 'Open Source Model', 
    color: '#a855f7',
    tier: 'open-source'
  },
]

/**
 * useVideoUrl - Fetches video/audio URLs with zero-trust verification
 * 
 * FALLBACK CHAIN:
 * 1. Lip-synced video (HeyGen/Fal/Sync/MuseTalk) - Best experience
 * 2. Base video + audio overlay - Good experience  
 * 3. Static image + TTS audio - Acceptable (generates on-demand)
 * 4. Static image only - Minimum viable
 */
export function useVideoUrl(
  dayOfYear: number,
  phase: LessonPhase,
  age: number,
  archetype: KellyArchetype,
  language: string = 'en',
  source: VideoSource = 'auto'
) {
  const params = new URLSearchParams({
    day: dayOfYear.toString(),
    phase,
    age: age.toString(),
    archetype,
    language,
    source, // NEW: Specify which engine to fetch from
  })

  const url = `/api/video/url?${params.toString()}`
  
  const { data, error, isLoading, mutate } = useSWR<VideoUrlResponse>(
    url,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 5000, // Reduced from 30s to 5s for faster updates
      revalidateOnMount: true,
    }
  )

  // Determine the best available content mode
  const hasVideo = !!data?.url
  const hasAudio = !!data?.audioUrl
  const hasFallback = !!data?.fallbackUrl
  
  // Audio URL from main source only (no auto TTS - that causes rate limits)
  const audioUrl = data?.audioUrl || null
  
  // Content mode for UI feedback
  const contentMode: 'video' | 'audio-only' | 'image-only' | 'loading' = 
    isLoading ? 'loading' :
    hasVideo ? 'video' :
    hasAudio ? 'audio-only' :
    'image-only'

  return {
    videoUrl: data?.url || null,
    audioUrl,
    fallbackUrl: data?.fallbackUrl || '/images/kelly-fallback.jpg',
    thumbnailUrl: data?.thumbnailUrl || null,
    status: data?.status || (isLoading ? 'generating' : 'ready'),
    contentMode,
    hasVideo,
    hasAudio,
    hasFallback,
    isLoading,
    error,
    refresh: mutate,
  }
}

// Get script for a specific phase
export function getPhaseScript(lesson: Lesson | null, phase: LessonPhase): string {
  if (!lesson) return ''
  
  switch (phase) {
    case 'hook':
      return lesson.hook_script || ''
    case 'story':
      return lesson.story_script || ''
    case 'wonder':
      return lesson.wonder_script || ''
    case 'action':
      return lesson.action_script || ''
    case 'wisdom':
      return lesson.wisdom_script || ''
    default:
      return ''
  }
}

// Phase display metadata
export const PHASE_METADATA: Record<LessonPhase, { label: string; icon: string; description: string }> = {
  hook: {
    label: 'Hook',
    icon: 'Sparkles',
    description: 'Capture your curiosity',
  },
  story: {
    label: 'Story',
    icon: 'BookOpen',
    description: 'The journey begins',
  },
  wonder: {
    label: 'Discover',
    icon: 'Lightbulb',
    description: 'Dive deeper',
  },
  action: {
    label: 'Try It',
    icon: 'Play',
    description: 'Put it into practice',
  },
  wisdom: {
    label: 'Reflect',
    icon: 'Quote',
    description: 'What we learned',
  },
}

// Phase order
export const PHASE_ORDER: LessonPhase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
