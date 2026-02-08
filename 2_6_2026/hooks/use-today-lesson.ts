'use client'

import useSWR from 'swr'
import { useMemo } from 'react'
import type { LessonPhase, KellyArchetype } from '@/lib/types'

// ============================================================================
// CINEMATIC KELLY — useTodayLesson Hook
// Fetches lesson content from /api/lesson/today with full language/tone/archetype support
// ============================================================================

export interface LessonPhaseData {
  phase: number
  phase_name: string
  content_text: string
  audio_url: string | null
  duration_seconds: number
}

export interface ArchetypeInfo {
  name: string
  hook: string | null
  emotion: string
}

export interface TodayLessonResponse {
  day_number: number
  date: string
  title: string
  subject: string
  theme: string
  universal_truth: string
  icon_emoji: string
  language: string
  tone: string
  phases: LessonPhaseData[]
  archetype: ArchetypeInfo
  hook_fact: string | null
  hook_correct_answer: boolean
  hook_question: string | null
  teaching_moment: string | null
  _sources: Record<string, boolean>
  _generated: string
}

// Map KellyArchetype to spec archetype names
const archetypeToTone: Record<string, string> = {
  storyteller: 'storyteller',
  explorer: 'explorer',
  scientist: 'scientist',
  architect: 'architect',
  strategist: 'strategist',
  diplomat: 'diplomat',
  mystic: 'mystic',
  rebel: 'rebel',
  macgyver: 'inventor',
  empath: 'empath',
  provider: 'provider',
  survivor: 'survivor',
}

// Map age number to tone parameter
function ageToTone(age: number): string {
  if (age <= 12) return 'kid'
  if (age >= 65) return 'elder'
  return 'mentor'
}

// Map phase name to phase number and vice versa
const PHASE_NAME_TO_NUM: Record<string, number> = {
  hook: 1, story: 2, wonder: 3, action: 4, wisdom: 5
}
const PHASE_NUM_TO_NAME: Record<number, LessonPhase> = {
  1: 'hook', 2: 'story', 3: 'wonder', 4: 'action', 5: 'wisdom'
}

const fetcher = async (url: string) => {
  try {
    const res = await fetch(url, {
      cache: 'no-store',
      headers: { 'Cache-Control': 'no-cache' }
    })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export function useTodayLesson(
  dayOfYear: number,
  options: {
    age?: number
    archetype?: KellyArchetype
    language?: string
  } = {}
) {
  const { age = 25, archetype = 'explorer', language = 'en' } = options
  
  const tone = ageToTone(age)
  const archetypeName = archetypeToTone[archetype] || archetype
  
  const params = new URLSearchParams({
    day: dayOfYear.toString(),
    lang: language,
    tone,
    archetype: archetypeName,
  })
  
  const swrKey = `/api/lesson/today?${params.toString()}`
  
  const { data, error, isLoading, mutate } = useSWR<TodayLessonResponse>(
    swrKey,
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnMount: true,
      dedupingInterval: 2000,
      keepPreviousData: false,
      refreshInterval: 0,
    }
  )
  
  // Extract scripts by phase name for compatibility with existing TikTokFeed props
  const phaseScripts = useMemo(() => {
    if (!data?.phases) return null
    const scripts: Record<string, string> = {}
    for (const p of data.phases) {
      const name = p.phase_name || PHASE_NUM_TO_NAME[p.phase]
      if (name && p.content_text) {
        scripts[name] = p.content_text
      }
    }
    return Object.keys(scripts).length > 0 ? scripts : null
  }, [data?.phases])
  
  // Get audio URL for a specific phase
  // Returns { audio_url, duration_seconds } to match TikTokFeed's PhaseAudioData interface
  const getPhaseAudio = useMemo(() => {
    if (!data?.phases) return () => null
    const audioMap: Record<string, { audio_url: string | null; duration_seconds: number }> = {}
    for (const p of data.phases) {
      const name = p.phase_name || PHASE_NUM_TO_NAME[p.phase]
      if (name && p.audio_url) {
        audioMap[name] = { audio_url: p.audio_url, duration_seconds: p.duration_seconds }
      }
    }
    return (phase: LessonPhase) => audioMap[phase] || null
  }, [data?.phases])
  
  // Get content text for a specific phase
  const getPhaseContent = useMemo(() => {
    if (!data?.phases) return () => ''
    const contentMap: Record<string, string> = {}
    for (const p of data.phases) {
      const name = p.phase_name || PHASE_NUM_TO_NAME[p.phase]
      if (name) {
        contentMap[name] = p.content_text
      }
    }
    return (phase: LessonPhase) => contentMap[phase] || ''
  }, [data?.phases])
  
  return {
    // Core lesson data
    lesson: data || null,
    title: data?.title || 'Loading...',
    subject: data?.subject || '',
    theme: data?.theme || '',
    universalTruth: data?.universal_truth || '',
    dayNumber: data?.day_number || dayOfYear,
    date: data?.date || '',
    
    // Phase data
    phases: data?.phases || [],
    phaseScripts,
    getPhaseAudio,
    getPhaseContent,
    
    // Archetype
    archetype: data?.archetype || { name: archetypeName, hook: null, emotion: 'warm' },
    
    // Hook / True-False game
    hookFact: data?.hook_fact || null,
    hookCorrectAnswer: data?.hook_correct_answer ?? true,
    hookQuestion: data?.hook_question || null,
    teachingMoment: data?.teaching_moment || null,
    
    // Data source info
    sources: data?._sources || {},
    
    // Loading state
    isLoading,
    error,
    refresh: mutate,
  }
}
