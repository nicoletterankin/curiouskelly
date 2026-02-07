// ============================================================================
// CURIOUS KELLY - DATE/TIME UTILITIES
// ============================================================================
// CRITICAL: All dates/times MUST be displayed in full format with timezone.
// NEVER use "Day X", abbreviated dates, or times without timezone.
// ============================================================================

// Locale mapping for i18n date formatting
export const DATE_LOCALE_MAP: Record<string, string> = {
  en: 'en-US',
  es: 'es-ES',
  fr: 'fr-FR',
  de: 'de-DE',
  pt: 'pt-BR',
  zh: 'zh-CN',
}

/**
 * Format a date as full weekday, month day, year
 * Example: "Friday, January 17, 2026" (en) / "Viernes, 17 de enero de 2026" (es)
 * NEVER returns: "Day 17", "1/17", "Jan 17"
 */
export function formatFullDate(
  date: Date,
  timezone: string,
  language: string = 'en'
): string {
  const locale = DATE_LOCALE_MAP[language] || 'en-US'
  return new Intl.DateTimeFormat(locale, {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: timezone,
  }).format(date)
}

/**
 * Format a short date (month + day)
 * Example: "Jan 17" (en) / "17 ene" (es)
 */
export function formatShortDate(
  date: Date,
  timezone: string,
  language: string = 'en'
): string {
  const locale = DATE_LOCALE_MAP[language] || 'en-US'
  return new Intl.DateTimeFormat(locale, {
    month: 'short',
    day: 'numeric',
    timeZone: timezone,
  }).format(date)
}

/**
 * Format time with timezone abbreviation
 * Example: "3:45 PM PST"
 * NEVER returns: "3:45", "15:45"
 */
export function formatFullTime(
  date: Date,
  timezone: string,
  locale: string = 'en-US'
): string {
  return new Intl.DateTimeFormat(locale, {
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
    timeZone: timezone,
  }).format(date)
}

/**
 * Format full date and time together
 * Example: "Friday, January 17, 2026 at 3:45 PM PST"
 */
export function formatFullDateTime(
  date: Date,
  timezone: string,
  locale: string = 'en-US'
): string {
  return `${formatFullDate(date, timezone, locale)} at ${formatFullTime(date, timezone, locale)}`
}

/**
 * Get timezone abbreviation (PST, EST, GMT, etc.)
 */
export function getTimezoneAbbreviation(
  timezone: string,
  date: Date = new Date()
): string {
  try {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      timeZoneName: 'short',
    }).formatToParts(date)
    return parts.find((p) => p.type === 'timeZoneName')?.value || timezone
  } catch {
    return 'UTC'
  }
}

/**
 * Get the day of year (1-366) for a date in a specific timezone
 */
export function getDayOfYear(date: Date, timezone: string): number {
  // Convert to timezone-local date string and parse
  const localDateStr = date.toLocaleDateString('en-CA', { timeZone: timezone })
  const [year, month, day] = localDateStr.split('-').map(Number)
  const localDate = new Date(year, month - 1, day)
  const startOfYear = new Date(year, 0, 1)
  const diff = localDate.getTime() - startOfYear.getTime()
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay) + 1
}

/**
 * Check if two dates are the same day in a timezone
 */
export function isSameDay(
  date1: Date,
  date2: Date,
  timezone: string
): boolean {
  const d1 = date1.toLocaleDateString('en-CA', { timeZone: timezone })
  const d2 = date2.toLocaleDateString('en-CA', { timeZone: timezone })
  return d1 === d2
}

/**
 * Check if a date is the user's birthday
 */
export function isBirthday(
  birthday: Date | null,
  today: Date,
  timezone: string
): boolean {
  if (!birthday) return false
  
  const todayLocal = today.toLocaleDateString('en-US', {
    timeZone: timezone,
    month: '2-digit',
    day: '2-digit',
  })
  const birthdayLocal = birthday.toLocaleDateString('en-US', {
    timeZone: timezone,
    month: '2-digit',
    day: '2-digit',
  })
  
  return todayLocal === birthdayLocal
}

/**
 * Format a countdown duration into human-readable text
 * Example: "2h 34m 12s"
 * NEVER returns: "02:34:12", "9252 seconds"
 */
export function formatCountdown(milliseconds: number): string {
  if (milliseconds <= 0) return 'Now'
  
  const seconds = Math.floor(milliseconds / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  
  const parts: string[] = []
  if (days > 0) parts.push(`${days}d`)
  if (hours % 24 > 0 || days > 0) parts.push(`${hours % 24}h`)
  if (minutes % 60 > 0 || hours > 0) parts.push(`${minutes % 60}m`)
  parts.push(`${seconds % 60}s`)
  
  return parts.join(' ')
}

/**
 * Get time-based greeting for the user
 */
export function getGreeting(date: Date, timezone: string): string {
  const hour = parseInt(
    date.toLocaleTimeString('en-US', {
      timeZone: timezone,
      hour: 'numeric',
      hour12: false,
    })
  )
  
  if (hour >= 5 && hour < 12) return 'Good morning'
  if (hour >= 12 && hour < 17) return 'Good afternoon'
  if (hour >= 17 && hour < 21) return 'Good evening'
  return 'Welcome back'
}

/**
 * Validate a timezone string
 */
export function isValidTimezone(tz: string): boolean {
  try {
    Intl.DateTimeFormat(undefined, { timeZone: tz })
    return true
  } catch {
    return false
  }
}

/**
 * Get the browser's detected timezone
 */
export function getBrowserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone
  } catch {
    return 'UTC'
  }
}

/**
 * Map age number to age variant (matches kelly_lesson_assets.age_group)
 * 3 primary variants with content: kid (<=12), adult (13-64), elder (65+)
 */
export function getAgeVariant(age: number): 'kid' | 'adult' | 'elder' {
  if (age <= 12) return 'kid'
  if (age >= 65) return 'elder'
  return 'adult'
}

/**
 * Get Kelly's display age based on user age preference
 * Matches the 3 age variants: kid, adult, senior
 */
export function getKellyAge(userAge: number): number {
  if (userAge <= 12) return 10    // kid Kelly
  if (userAge >= 65) return 68    // senior Kelly
  return 30                        // adult Kelly
}
