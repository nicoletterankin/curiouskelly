'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import useSWR, { mutate } from 'swr'

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
type Phase = typeof PHASES[number]

interface PhaseCompleteResponse {
  success: boolean
  phase: string
  dayOfYear: number
  completedPhases: string[]
  dayComplete: boolean
  streak: {
    current: number
    longest: number
    updated: boolean
  }
}

interface DayProgressResponse {
  completedPhases: string[]
  dayComplete: boolean
}

interface StreakData {
  currentStreak: number
  longestStreak: number
  lastLessonDate: string | null
  completedDaysThisYear: number[]
}

const fetcher = async (url: string) => {
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch')
  return res.json()
}

export function useProgress(dayOfYear: number) {
  const [isCompleting, setIsCompleting] = useState(false)
  const [showCelebration, setShowCelebration] = useState(false)
  const hasMigrated = useRef(false)

  // Fetch current day's progress
  const { data: dayProgress, mutate: mutateDayProgress } = useSWR<DayProgressResponse>(
    dayOfYear ? `/api/learner/phase-complete?day=${dayOfYear}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 5000,
    }
  )

  // Fetch overall streak data
  const { data: streakData, mutate: mutateStreak } = useSWR<StreakData>(
    '/api/learner/streak',
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 10000,
    }
  )

  // Migrate localStorage data on first load (one-time)
  useEffect(() => {
    if (hasMigrated.current) return
    hasMigrated.current = true

    const migrateLocalStorage = async () => {
      try {
        // Check for existing localStorage data
        const localStreak = parseInt(localStorage.getItem('kelly_streak') || '0', 10)
        const localCompletedStr = localStorage.getItem('kelly_completed_days')
        const localCompletedDays = localCompletedStr ? JSON.parse(localCompletedStr) : []

        if (localStreak > 0 || localCompletedDays.length > 0) {
          const res = await fetch('/api/learner/streak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ localStreak, localCompletedDays })
          })

          if (res.ok) {
            const result = await res.json()
            if (result.migrated) {
              // Clear localStorage after successful migration
              localStorage.removeItem('kelly_streak')
              localStorage.removeItem('kelly_completed_days')
              localStorage.setItem('kelly_streak_migrated', 'true')
              // Refresh streak data
              mutateStreak()
            }
          }
        }
      } catch (e) {
        console.warn('[useProgress] Migration failed:', e)
      }
    }

    // Only migrate if not already migrated
    if (!localStorage.getItem('kelly_streak_migrated')) {
      migrateLocalStorage()
    }
  }, [mutateStreak])

  // Complete a phase
  const completePhase = useCallback(async (
    phase: Phase,
    timeSpentSeconds: number = 0
  ): Promise<PhaseCompleteResponse | null> => {
    if (isCompleting) return null
    setIsCompleting(true)

    try {
      const res = await fetch('/api/learner/phase-complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dayOfYear,
          phase,
          timeSpentSeconds
        })
      })

      if (!res.ok) {
        throw new Error('Failed to complete phase')
      }

      const result: PhaseCompleteResponse = await res.json()

      // Update local cache
      mutateDayProgress({
        completedPhases: result.completedPhases,
        dayComplete: result.dayComplete
      }, false)

      // If day is complete, show celebration and update streak
      if (result.dayComplete) {
        setShowCelebration(true)
        mutateStreak()
        // Also invalidate the progress endpoint used by stats widgets
        mutate('/api/learner/progress')
      }

      return result
    } catch (error) {
      console.error('[useProgress] completePhase error:', error)
      return null
    } finally {
      setIsCompleting(false)
    }
  }, [dayOfYear, isCompleting, mutateDayProgress, mutateStreak])

  // Check if a specific phase is completed
  const isPhaseCompleted = useCallback((phase: Phase): boolean => {
    return dayProgress?.completedPhases?.includes(phase) ?? false
  }, [dayProgress])

  // Get completion percentage for the day
  const getDayCompletionPercent = useCallback((): number => {
    const completed = dayProgress?.completedPhases?.length ?? 0
    return Math.round((completed / PHASES.length) * 100)
  }, [dayProgress])

  // Dismiss celebration
  const dismissCelebration = useCallback(() => {
    setShowCelebration(false)
  }, [])

  return {
    // Day progress
    completedPhases: dayProgress?.completedPhases ?? [],
    dayComplete: dayProgress?.dayComplete ?? false,
    completionPercent: getDayCompletionPercent(),
    
    // Streak data
    currentStreak: streakData?.currentStreak ?? 0,
    longestStreak: streakData?.longestStreak ?? 0,
    completedDaysThisYear: streakData?.completedDaysThisYear ?? [],
    
    // Actions
    completePhase,
    isPhaseCompleted,
    
    // UI state
    isCompleting,
    showCelebration,
    dismissCelebration,
    
    // Refresh functions
    refreshDayProgress: mutateDayProgress,
    refreshStreak: mutateStreak
  }
}

// Hook to get just streak data (for widgets that don't need full progress)
export function useStreak() {
  const { data, mutate: mutateStreak } = useSWR<StreakData>(
    '/api/learner/streak',
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 10000,
    }
  )

  return {
    currentStreak: data?.currentStreak ?? 0,
    longestStreak: data?.longestStreak ?? 0,
    completedDaysThisYear: data?.completedDaysThisYear ?? [],
    refreshStreak: mutateStreak
  }
}
