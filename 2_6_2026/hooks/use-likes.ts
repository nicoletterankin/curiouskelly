'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import useSWR from 'swr'
import { getFingerprint } from '@/lib/fingerprint'

interface LikesResponse {
  count: number
  userLiked: boolean
}

// KELLY UI RULE: NO FAKE DATA - return 0 when no real data available
const fetcher = async (url: string): Promise<LikesResponse> => {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000)
    
    const res = await fetch(url, { signal: controller.signal })
    clearTimeout(timeout)
    
    if (!res.ok) {
      return { count: 0, userLiked: false } // Real zero, not fake number
    }
    return res.json()
  } catch {
    return { count: 0, userLiked: false } // Real zero on error
  }
}

export function useLikes(day: number) {
  const [visitorId, setVisitorId] = useState<string>('')
  const [localLiked, setLocalLiked] = useState<boolean | null>(null)
  const [localCount, setLocalCount] = useState<number | null>(null)
  const pendingRef = useRef(false)
  const retryQueueRef = useRef<Array<{ day: number; visitorId: string; liked: boolean }>>([])
  
  useEffect(() => {
    setVisitorId(getFingerprint())
    
    // Restore local state from sessionStorage
    const stored = sessionStorage.getItem(`like_${day}`)
    if (stored) {
      const { liked, count } = JSON.parse(stored)
      setLocalLiked(liked)
      setLocalCount(count)
    }
  }, [day])
  
  const { data, error, isLoading, mutate } = useSWR<LikesResponse>(
    visitorId ? `/api/likes?day=${day}&visitorId=${visitorId}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000, // 1 minute deduping
      errorRetryCount: 2,
      errorRetryInterval: 3000,
      fallbackData: { count: 0, userLiked: false },
    }
  )
  
  // Process retry queue when online
  useEffect(() => {
    const processQueue = async () => {
      if (retryQueueRef.current.length === 0 || pendingRef.current) return
      
      pendingRef.current = true
      const item = retryQueueRef.current[0]
      
      try {
        const res = await fetch('/api/likes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ day: item.day, visitorId: item.visitorId })
        })
        
        if (res.ok) {
          retryQueueRef.current.shift() // Remove successful item
        }
      } catch {
        // Keep in queue for retry
      } finally {
        pendingRef.current = false
      }
    }
    
    const interval = setInterval(processQueue, 10000)
    return () => clearInterval(interval)
  }, [])
  
  const toggleLike = useCallback(async () => {
    if (!visitorId) return
    
    // Optimistic update with local state
    const currentLiked = localLiked ?? data?.userLiked ?? false
    const currentCount = localCount ?? data?.count ?? 0
    const newLiked = !currentLiked
    const newCount = Math.max(0, currentCount + (newLiked ? 1 : -1))
    
    setLocalLiked(newLiked)
    setLocalCount(newCount)
    
    // Persist to sessionStorage for offline support
    sessionStorage.setItem(`like_${day}`, JSON.stringify({ liked: newLiked, count: newCount }))
    
    // Optimistic SWR update
    mutate({ count: newCount, userLiked: newLiked }, false)
    
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 5000)
      
      const res = await fetch('/api/likes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ day, visitorId }),
        signal: controller.signal
      })
      
      clearTimeout(timeout)
      
      if (res.ok) {
        const result = await res.json()
        mutate(result)
        setLocalCount(result.count)
      }
    } catch {
      // Add to retry queue for later
      retryQueueRef.current.push({ day, visitorId, liked: newLiked })
      // Keep optimistic state - user sees their action worked
    }
  }, [visitorId, day, localLiked, localCount, data, mutate])
  
  // KELLY UI RULE: Use local state if available, otherwise real data or 0
  const displayCount = localCount ?? data?.count ?? 0
  const displayLiked = localLiked ?? data?.userLiked ?? false
  
  return {
    count: displayCount,
    userLiked: displayLiked,
    toggleLike,
    isLoading: isLoading && localLiked === null,
    error,
    isOffline: !!error && localLiked !== null
  }
}
