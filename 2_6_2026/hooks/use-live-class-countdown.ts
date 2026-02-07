'use client'

import { useState, useEffect } from 'react'
import useSWR from 'swr'
import { formatFullDateTime, formatCountdown } from '@/lib/date-utils'
import type { LiveClassRoom, LiveClassCountdown, AgeVariant } from '@/lib/types'

interface LiveClassResponse {
  id: string
  day: number
  age_bracket: string
  scheduled_at: string
  started_at: string | null
  ended_at: string | null
  status: 'scheduled' | 'starting' | 'live' | 'ended'
  participant_count: number
  max_participants: number
  kelly_present: boolean
}

const fetcher = async (url: string): Promise<LiveClassResponse | null> => {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000) // 5s timeout
    
    const res = await fetch(url, { signal: controller.signal })
    clearTimeout(timeout)
    
    if (!res.ok) {
      // On 404, 429, or any error, return null gracefully
      return null
    }
    const text = await res.text()
    if (!text || text === 'null') return null
    
    try {
      return JSON.parse(text)
    } catch {
      return null
    }
  } catch {
    // Network error or timeout - fail silently
    return null
  }
}

export function useLiveClassCountdown(
  ageVariant?: AgeVariant,
  timezone: string = 'UTC',
  locale: string = 'en-US'
): LiveClassCountdown {
  const [now, setNow] = useState(new Date())

  // Build API URL with optional age filter
  const apiUrl = ageVariant
    ? `/api/live-classes/next?age=${ageVariant}`
    : '/api/live-classes/next'

  // Fetch next scheduled class
  const { data: classData, error, isLoading } = useSWR<LiveClassResponse | null>(
    apiUrl,
    fetcher,
    { refreshInterval: 30000 } // Refresh every 30 seconds
  )

  // Tick every second for real-time countdown
  useEffect(() => {
    const interval = setInterval(() => {
      setNow(new Date())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  // No upcoming class
  if (!classData) {
    return {
      class: null,
      days: 0,
      hours: 0,
      minutes: 0,
      seconds: 0,
      isStartingSoon: false,
      isLive: false,
      hasEnded: false,
      noUpcoming: true,
      countdownText: 'No upcoming classes',
      fullText: 'No live classes scheduled',
      scheduledAtFormatted: '',
      isLoading,
      error: error || null,
    }
  }

  // Parse scheduled time
  const scheduledAt = new Date(classData.scheduled_at)
  const diff = scheduledAt.getTime() - now.getTime()

  // Format the scheduled time (ALWAYS full format with timezone)
  const scheduledAtFormatted = formatFullDateTime(scheduledAt, timezone, locale)

  // Convert response to LiveClassRoom type
  const liveClass: LiveClassRoom = {
    id: classData.id,
    day: classData.day,
    age_bracket: classData.age_bracket,
    scheduled_at: scheduledAt,
    started_at: classData.started_at ? new Date(classData.started_at) : null,
    ended_at: classData.ended_at ? new Date(classData.ended_at) : null,
    status: classData.status,
    participant_count: classData.participant_count,
    max_participants: classData.max_participants,
    kelly_present: classData.kelly_present,
    created_at: new Date(),
  }

  // Class is live
  if (classData.status === 'live' || (diff <= 0 && classData.status !== 'ended')) {
    return {
      class: liveClass,
      days: 0,
      hours: 0,
      minutes: 0,
      seconds: 0,
      isStartingSoon: false,
      isLive: true,
      hasEnded: false,
      noUpcoming: false,
      countdownText: 'LIVE NOW',
      fullText: `Live class is happening now with ${classData.participant_count} learners`,
      scheduledAtFormatted,
      isLoading: false,
      error: null,
    }
  }

  // Class has ended
  if (classData.status === 'ended') {
    return {
      class: liveClass,
      days: 0,
      hours: 0,
      minutes: 0,
      seconds: 0,
      isStartingSoon: false,
      isLive: false,
      hasEnded: true,
      noUpcoming: false,
      countdownText: 'Ended',
      fullText: 'This class has ended',
      scheduledAtFormatted,
      isLoading: false,
      error: null,
    }
  }

  // Calculate countdown components
  const totalSeconds = Math.floor(diff / 1000)
  const days = Math.floor(totalSeconds / (60 * 60 * 24))
  const hours = Math.floor((totalSeconds % (60 * 60 * 24)) / (60 * 60))
  const minutes = Math.floor((totalSeconds % (60 * 60)) / 60)
  const seconds = totalSeconds % 60

  // Format countdown text
  const countdownText = formatCountdown(diff)

  // Check if starting soon (< 5 minutes)
  const isStartingSoon = diff < 5 * 60 * 1000 && diff > 0

  return {
    class: liveClass,
    days,
    hours,
    minutes,
    seconds,
    isStartingSoon,
    isLive: false,
    hasEnded: false,
    noUpcoming: false,
    countdownText,
    fullText: `Next live class starts in ${countdownText}`,
    scheduledAtFormatted,
    isLoading: false,
    error: null,
  }
}
