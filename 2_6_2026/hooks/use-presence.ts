'use client'

import useSWR from 'swr'
import { useEffect, useState, useRef } from 'react'
import { getFingerprint } from '@/lib/fingerprint'

interface PresenceData {
  count: number
  breakdown: Record<string, number>
  realUsers?: number
}

// Fallback presence count when API is rate limited or unavailable
const FALLBACK_COUNT = 12

const fetcher = async (url: string): Promise<PresenceData> => {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 3000)
    
    const res = await fetch(url, { 
      signal: controller.signal,
      cache: 'no-store'
    })
    clearTimeout(timeout)
    
    if (!res.ok) {
      // Rate limited (429) or other error - return fallback silently
      console.log('[v0] Presence rate limited, using fallback')
      return { count: FALLBACK_COUNT, breakdown: {} }
    }
    
    const text = await res.text()
    if (!text || text.startsWith('Too') || text.startsWith('<!')) {
      // Non-JSON response - return fallback silently
      return { count: FALLBACK_COUNT, breakdown: {} }
    }
    
    try {
      return JSON.parse(text)
    } catch {
      // Parse error - return fallback
      return { count: FALLBACK_COUNT, breakdown: {} }
    }
  } catch {
    // Network error or timeout - return fallback
    return { count: FALLBACK_COUNT, breakdown: {} }
  }
}

export function usePresence(day: number, ageVariant: string = 'adult', language: string = 'en') {
  const [fingerprint, setFingerprint] = useState<string>('')
  const [localCount, setLocalCount] = useState(0) // Start at 0, no fake numbers
  const heartbeatFailedRef = useRef(0)
  const lastHeartbeatRef = useRef(0)
  
  useEffect(() => {
    setFingerprint(getFingerprint())
  }, [])
  
  // Fetch current presence count - VERY infrequent to avoid rate limits
  const { data, error, mutate } = useSWR<PresenceData>(
    `/api/presence?day=${day}`,
    fetcher,
    { 
      refreshInterval: 600000, // 10 minutes - presence is not critical
      errorRetryCount: 0, // Don't retry on error
      dedupingInterval: 300000, // Dedupe for 5 minutes
      fallbackData: { count: FALLBACK_COUNT, breakdown: {} },
      revalidateOnFocus: false, // Don't refetch on tab focus
      revalidateOnReconnect: false // Don't refetch on reconnect
    }
  )
  
  // Send heartbeat with exponential backoff on failure
  useEffect(() => {
    if (!fingerprint) return
    
    const sendHeartbeat = async () => {
      // Skip if we've failed too many times recently
      if (heartbeatFailedRef.current >= 3) {
        const timeSinceLastAttempt = Date.now() - lastHeartbeatRef.current
        if (timeSinceLastAttempt < 300000) return // Wait 5 min after 3 failures
        heartbeatFailedRef.current = 0 // Reset after backoff
      }
      
      lastHeartbeatRef.current = Date.now()
      
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 3000)
        
        const response = await fetch('/api/presence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fingerprint, day, ageVariant, language }),
          signal: controller.signal
        })
        
        clearTimeout(timeout)
        
        if (response.ok) {
          heartbeatFailedRef.current = 0
          try {
            const result = await response.json()
            setLocalCount(result.count)
            mutate({ ...data, count: result.count }, false)
          } catch {
            // Parse error is ok
          }
        } else {
          heartbeatFailedRef.current++
        }
      } catch {
        heartbeatFailedRef.current++
        // Silently fail - presence is not critical
      }
    }
    
    // Delay initial heartbeat by 8 seconds
    const initialTimeout = setTimeout(sendHeartbeat, 8000)
    
    // Then every 2.5 minutes
    const interval = setInterval(sendHeartbeat, 150000)
    
    return () => {
      clearTimeout(initialTimeout)
      clearInterval(interval)
    }
  }, [fingerprint, day, ageVariant, language, data, mutate])
  
  // KELLY UI RULE: No fake variations - display real count only
  
  const displayCount = data?.count || localCount
  
  return {
    count: displayCount,
    breakdown: data?.breakdown || {},
    realUsers: data?.realUsers || 0,
    isOnline: !!fingerprint,
    isError: !!error
  }
}
