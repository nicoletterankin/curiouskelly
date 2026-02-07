'use client'

import { useState, useEffect, useCallback } from 'react'

interface LessonCacheStatus {
  day: number
  cached: boolean
  phases: {
    hook: boolean
    story: boolean
    wonder: boolean
    action: boolean
    wisdom: boolean
  }
}

export function useOfflineLessons() {
  const [isOnline, setIsOnline] = useState(true)
  const [isServiceWorkerReady, setIsServiceWorkerReady] = useState(false)
  const [cachedLessons, setCachedLessons] = useState<number[]>([])

  // Monitor online status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    setIsOnline(navigator.onLine)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Check service worker status
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(() => {
        setIsServiceWorkerReady(true)
      })
    }
  }, [])

  // Prefetch a lesson for offline use
  const prefetchLesson = useCallback(async (
    day: number,
    ageGroup = 'adult',
    language = 'en',
    archetype = 'explorer'
  ) => {
    if (!isServiceWorkerReady) {
      console.warn('[OfflineLessons] Service worker not ready')
      return false
    }

    try {
      const registration = await navigator.serviceWorker.ready
      registration.active?.postMessage({
        type: 'PREFETCH_LESSON',
        day,
        ageGroup,
        language,
        archetype,
      })

      // Track that we've requested this lesson to be cached
      setCachedLessons((prev) => {
        if (!prev.includes(day)) {
          const updated = [...prev, day].sort((a, b) => a - b)
          localStorage.setItem('kelly_cached_lessons', JSON.stringify(updated))
          return updated
        }
        return prev
      })

      console.log('[OfflineLessons] Prefetch requested for day', day)
      return true
    } catch (err) {
      console.error('[OfflineLessons] Prefetch failed:', err)
      return false
    }
  }, [isServiceWorkerReady])

  // Prefetch a range of lessons (e.g., next 7 days)
  const prefetchLessonRange = useCallback(async (
    startDay: number,
    count: number,
    ageGroup = 'adult',
    language = 'en',
    archetype = 'explorer'
  ) => {
    const promises = []
    for (let i = 0; i < count; i++) {
      const day = ((startDay - 1 + i) % 365) + 1 // Wrap around at 365
      promises.push(prefetchLesson(day, ageGroup, language, archetype))
    }
    await Promise.all(promises)
    console.log('[OfflineLessons] Prefetched', count, 'lessons starting from day', startDay)
  }, [prefetchLesson])

  // Check if a specific lesson is cached
  const isLessonCached = useCallback((day: number): boolean => {
    return cachedLessons.includes(day)
  }, [cachedLessons])

  // Clear all cached lessons
  const clearCache = useCallback(async () => {
    if (!isServiceWorkerReady) return

    try {
      const registration = await navigator.serviceWorker.ready
      registration.active?.postMessage({ type: 'CLEAR_CACHES' })
      setCachedLessons([])
      localStorage.removeItem('kelly_cached_lessons')
      console.log('[OfflineLessons] Cache cleared')
    } catch (err) {
      console.error('[OfflineLessons] Clear cache failed:', err)
    }
  }, [isServiceWorkerReady])

  // Load cached lessons from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('kelly_cached_lessons')
    if (stored) {
      try {
        setCachedLessons(JSON.parse(stored))
      } catch {
        // Invalid JSON, reset
        localStorage.removeItem('kelly_cached_lessons')
      }
    }
  }, [])

  return {
    isOnline,
    isServiceWorkerReady,
    cachedLessons,
    prefetchLesson,
    prefetchLessonRange,
    isLessonCached,
    clearCache,
  }
}

// Get current day of year
export function getDayOfYear(): number {
  const now = new Date()
  const start = new Date(now.getFullYear(), 0, 0)
  const diff = now.getTime() - start.getTime()
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay)
}
