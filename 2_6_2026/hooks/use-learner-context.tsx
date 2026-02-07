'use client'

import { useState, useEffect, useCallback } from 'react'
import useSWR from 'swr'
import {
  formatFullDate,
  formatFullTime,
  formatFullDateTime,
  getTimezoneAbbreviation,
  getDayOfYear,
  isBirthday,
  getGreeting,
  getBrowserTimezone,
} from '@/lib/date-utils'
import type { LearnerContext, KellyArchetype } from '@/lib/types'

interface UserProfileResponse {
  id: string
  name: string | null
  email: string
  birthday: string | null
  preferred_timezone: string | null
  preferred_language: string
  preferred_avatar_age: number
  preferred_archetype: KellyArchetype
  subscription_status: string | null
}

const fetcher = async (url: string) => {
  const res = await fetch(url)
  if (!res.ok) {
    if (res.status === 401) return null // Not authenticated
    throw new Error('Failed to fetch user profile')
  }
  return res.json()
}

export function useLearnerContext(): LearnerContext {
  // Local state for time-sensitive data
  const [timezone, setTimezone] = useState<string>('UTC')
  const [locale, setLocale] = useState<string>('en-US')
  const [now, setNow] = useState<Date>(new Date())
  const [isClient, setIsClient] = useState(false)
  const [dayOverride, setDayOverride] = useState<number | null>(null)

  // Fetch user profile from API
  const { data: user, error, isLoading } = useSWR<UserProfileResponse | null>(
    '/api/user/profile',
    fetcher,
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      dedupingInterval: 30000,
    }
  )

  // Detect timezone and locale on client mount
  useEffect(() => {
    setIsClient(true)
    const detectedTz = getBrowserTimezone()
    const detectedLocale = navigator.language || 'en-US'

    // Use user's preferred timezone if set, otherwise use browser's
    setTimezone(user?.preferred_timezone || detectedTz)
    setLocale(detectedLocale)
    
    // Read ?day=N URL parameter for demo/testing only
    const params = new URLSearchParams(window.location.search)
    const urlDay = params.get('day')
    if (urlDay !== null) {
      const parsed = parseInt(urlDay, 10)
      if (!isNaN(parsed) && parsed >= 0 && parsed <= 365) {
        setDayOverride(parsed)
      }
    }
    // Otherwise use auto-calculated day from current date
  }, [user?.preferred_timezone])

  // Update `now` every second for accurate time display
  useEffect(() => {
    const interval = setInterval(() => {
      setNow(new Date())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  // Parse birthday if available
  const birthdayDate = user?.birthday ? new Date(user.birthday) : null

  // Calculate derived values - allow override for navigation
  // UPDATED: Day 34 now has 45 HeyGen videos! Removing cap.
  const rawCalculatedDay = isClient ? getDayOfYear(now, timezone) : 1
  const dayOfYear = dayOverride !== null ? dayOverride : rawCalculatedDay
  const isBirthdayToday = isClient && birthdayDate ? isBirthday(birthdayDate, now, timezone) : false

  // Format display strings
  const formattedDate = isClient ? formatFullDate(now, timezone, locale) : 'Loading...'
  const formattedTime = isClient ? formatFullTime(now, timezone, locale) : ''
  const formattedDateTime = isClient ? formatFullDateTime(now, timezone, locale) : 'Loading...'
  const timezoneAbbr = isClient ? getTimezoneAbbreviation(timezone, now) : ''
  const greeting = isClient ? getGreeting(now, timezone) : 'Welcome'

  return {
    // Identity
    userId: user?.id || null,
    name: user?.name || null,
    displayName: user?.name || 'Curious Learner',
    email: user?.email || null,
    birthday: birthdayDate,
    isBirthday: isBirthdayToday,
    isAuthenticated: !!user,

    // Temporal
    timezone,
    locale,
    now,
    dayOfYear,

    // Formatted strings (ALWAYS full format)
    formattedDate,
    formattedTime,
    formattedDateTime,
    timezoneAbbr,
    greeting,

    // Preferences
    preferredAge: user?.preferred_avatar_age || 30,
    preferredLanguage: user?.preferred_language || 'en',
    preferredArchetype: user?.preferred_archetype || 'storyteller',

    // Loading states
    isLoading: isLoading && !isClient,
    error: error || null,
    
    // Day navigation
    setDayOverride,
  }
}

// Export a context provider version for sharing state
import { createContext, useContext, type ReactNode } from 'react'

const LearnerContextContext = createContext<LearnerContext | null>(null)

export function LearnerContextProvider({ children }: { children: ReactNode }) {
  const context = useLearnerContext()
  return (
    <LearnerContextContext.Provider value={context}>
      {children}
    </LearnerContextContext.Provider>
  )
}

// Default context for SSR/prerendering when no provider is present
const defaultContext: LearnerContext = {
  userId: null,
  name: null,
  displayName: 'Curious Learner',
  email: null,
  birthday: null,
  isBirthday: false,
  isAuthenticated: false,
  timezone: 'UTC',
  locale: 'en-US',
  now: new Date(),
  dayOfYear: 1,
  formattedDate: 'Loading...',
  formattedTime: '',
  formattedDateTime: 'Loading...',
  timezoneAbbr: '',
  greeting: 'Welcome',
  preferredAge: 30,
  preferredLanguage: 'en',
  preferredArchetype: 'storyteller',
  isLoading: true,
  error: null,
  setDayOverride: () => {},
}

// Safe hook that never throws - returns default during SSR prerender
export function useLearner(): LearnerContext {
  const context = useContext(LearnerContextContext)
  // CRITICAL: Return default context during SSR/prerendering - NEVER throw
  // This allows Next.js to prerender pages that use this hook
  if (!context) {
    return defaultContext
  }
  return context
}
