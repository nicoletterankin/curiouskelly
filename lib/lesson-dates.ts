/**
 * Curious Kelly Lesson Date Utilities
 *
 * Canonical "Topic of the Day" calendar.
 *
 * - Day numbers are 1–365 and correspond to the traditional Gregorian
 *   calendar with **Day 1 = January 1, 2026**.
 * - The mapping is topic-based, not asset-based: January 1 is always
 *   topic 1, January 2 is topic 2, ..., December 31 is topic 365 —
 *   across all years, past and future.
 * - Leap years are handled by compressing after Feb 29 so that the same
 *   month/day (except Feb 29) always maps to the same topic.
 *
 * Leap year rules (canonical mapping):
 * - Non-leap years:  topic_day = day_of_year (1–365)
 * - Leap years:
 *     - Jan 1 – Feb 28: topic_day = day_of_year
 *     - Feb 29:         topic_day = 366 (special bonus lesson)
 *     - Mar 1 – Dec 31: topic_day = day_of_year - 1  (compress after Feb 29)
 *
 * User-facing content should ALWAYS show real dates.
 * The day_number is internal only (database, APIs, URLs, scheduling).
 */

// Calendar helpers

/**
 * Utility: check if a year is a leap year in the proleptic Gregorian calendar.
 */
function isLeapYear(year: number): boolean {
  if (year % 4 !== 0) return false;
  if (year % 100 !== 0) return true;
  return year % 400 === 0;
}

const DAYS_IN_MONTH_COMMON = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const DAYS_IN_MONTH_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

function getNextLeapYear(fromYear: number): number {
  let y = fromYear;
  while (!isLeapYear(y)) y++;
  return y;
}

function getDaysInMonth(year: number, month: number): number {
  return (isLeapYear(year) ? DAYS_IN_MONTH_LEAP : DAYS_IN_MONTH_COMMON)[month];
}

/**
 * Convert a (year, month, day) triple to a 1-based day-of-year value.
 * Month is 0-indexed (0 = January).
 */
function getDayOfYear(year: number, month: number, day: number): number {
  let doy = day;
  const daysInMonths = isLeapYear(year) ? DAYS_IN_MONTH_LEAP : DAYS_IN_MONTH_COMMON;
  for (let m = 0; m < month; m++) {
    doy += daysInMonths[m];
  }
  return doy;
}

/**
 * Convert a 1-based day-of-year to a Date in the given year.
 */
function dateFromDayOfYear(year: number, dayOfYear: number): Date {
  const leap = isLeapYear(year);
  const maxDay = leap ? 366 : 365;
  if (dayOfYear < 1 || dayOfYear > maxDay) {
    throw new Error(`Invalid dayOfYear ${dayOfYear} for year ${year}`);
  }

  const daysInMonths = leap ? DAYS_IN_MONTH_LEAP : DAYS_IN_MONTH_COMMON;
  let remaining = dayOfYear;
  let month = 0;
  while (month < 12 && remaining > daysInMonths[month]) {
    remaining -= daysInMonths[month];
    month++;
  }

  return new Date(year, month, remaining);
}

/**
 * Convert internal day_number (1–365) to a Date in a specific year.
 * 
 * In leap years, day numbers ≥ 60 are shifted by +1 day to account
 * for Feb 29 as an extra calendar day that does NOT get its own topic.
 * 
 * @param dayNumber - Internal lesson number (1–365)
 * @param year - Optional year override (defaults to the current year)
 * @returns Date object for that lesson in the specified year
 */
export function dayNumberToDate(dayNumber: number, year?: number): Date {
  if (dayNumber < 1 || dayNumber > 366) {
    throw new Error(`Invalid day number: ${dayNumber}. Must be 1–366.`);
  }

  const baseYear = year ?? new Date().getFullYear();

  // Leap Day is a special bonus lesson and maps to Feb 29.
  // If the caller asks for 366 in a non-leap year, map it to the next leap year
  // so the returned Date is valid.
  if (dayNumber === 366) {
    const targetYear = isLeapYear(baseYear) ? baseYear : getNextLeapYear(baseYear);
    return new Date(targetYear, 1, 29);
  }

  const leap = isLeapYear(baseYear);

  // Map canonical topic day → actual day-of-year for the target year.
  // In leap years, everything from Mar 1 onward moves one day later.
  const dayOfYear = leap && dayNumber >= 60 ? dayNumber + 1 : dayNumber;

  return dateFromDayOfYear(baseYear, dayOfYear);
}

/**
 * Convert a Date (assumed to already be in the relevant local timezone)
 * to the internal day_number (1–365) following the canonical mapping.
 * 
 * @param date - Calendar date in the user's local timezone
 * @returns Internal topic day number (1–365)
 */
export function dateToLessonDay(date: Date): number {
  const year = date.getFullYear();
  const month = date.getMonth();
  const day = date.getDate();

  const leap = isLeapYear(year);
  const dayOfYear = getDayOfYear(year, month, day); // 1–365/366

  if (!leap) {
    return dayOfYear;
  }

  // Leap year rules:
  // - Jan 1–Feb 28: canonical topic day == day-of-year
  // - Feb 29: special bonus lesson (366)
  // - Mar 1 onward: subtract 1 to keep month/day stable across years
  if (month === 1 && day === 29) {
    return 366;
  }

  if (dayOfYear <= 59) {
    // Jan 1–Feb 28
    return dayOfYear;
  }

  // March 1 onward: shift back by one
  return dayOfYear - 1;
}

/**
 * Canonical lesson identity for a calendar date.
 * - For most dates, `day` is 1–365 and `isLeapDay` is false.
 * - For Feb 29, `day` is 366 and `isLeapDay` is true.
 */
export function getCanonicalLessonDay(date: Date = new Date()): { day: number; isLeapDay: boolean } {
  const day = dateToLessonDay(date);
  return { day, isLeapDay: day === 366 };
}

/**
 * Canonical DB lookup key for core_lessons by month/day.
 * Feb 29 is represented as { isLeapDay: true }.
 */
export function getCanonicalLessonCalendarKey(
  date: Date = new Date()
): { calendarMonth: number; calendarDay: number; isLeapDay: boolean } {
  const { isLeapDay } = getCanonicalLessonDay(date);
  if (isLeapDay) {
    return { calendarMonth: 2, calendarDay: 29, isLeapDay: true };
  }

  return {
    calendarMonth: date.getMonth() + 1,
    calendarDay: date.getDate(),
    isLeapDay: false,
  };
}

/**
 * Get today's lesson day number in the local environment timezone.
 * 
 * For user-precise behavior across timezones, prefer
 * `getLessonDayForTimeZone` instead of this helper.
 * 
 * @returns Today's internal day number (1–365)
 */
export function getTodayLessonDay(): number {
  return dateToLessonDay(new Date());
}

/**
 * Get the lesson day number for a specific instant in time as experienced
 * in a particular IANA timezone (e.g. "America/New_York").
 * 
 * This is the recommended API for anything user-facing or global.
 * 
 * @param utcMillis - UTC timestamp in milliseconds (defaults to now)
 * @param timeZone - IANA timezone string (e.g. "America/Los_Angeles")
 */
export function getLessonDayForTimeZone(
  utcMillis: number = Date.now(),
  timeZone: string = 'UTC'
): number {
  const instant = new Date(utcMillis);

  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });

  const parts = formatter.formatToParts(instant);
  const lookup: Record<string, string> = {};
  for (const part of parts) {
    if (part.type !== 'literal') {
      lookup[part.type] = part.value;
    }
  }

  const year = Number(lookup.year);
  const month = Number(lookup.month) - 1; // 0-indexed
  const day = Number(lookup.day);

  const localDate = new Date(year, month, day);
  return dateToLessonDay(localDate);
}

/**
 * Format a date for user display
 * 
 * @param date - Date to format
 * @param options - Formatting options
 * @returns Formatted date string (e.g., "December 17")
 */
export function formatDateForDisplay(
  date: Date,
  options: {
    includeYear?: boolean;
    includeWeekday?: boolean;
    shortMonth?: boolean;
  } = {}
): string {
  const { includeYear = false, includeWeekday = false, shortMonth = false } = options;
  
  const formatOptions: Intl.DateTimeFormatOptions = {
    month: shortMonth ? 'short' : 'long',
    day: 'numeric'
  };
  
  if (includeYear) {
    formatOptions.year = 'numeric';
  }
  
  if (includeWeekday) {
    formatOptions.weekday = 'long';
  }
  
  return date.toLocaleDateString('en-US', formatOptions);
}

/**
 * Format today's lesson for user display
 * 
 * @returns Formatted date string for today's lesson
 */
export function formatTodayForDisplay(): string {
  const today = new Date();
  return formatDateForDisplay(today);
}

/**
 * Get the lesson date formatted for notifications/emails
 * 
 * @param dayNumber - Internal day number
 * @returns Object with formatted date strings
 */
export function getLessonDateStrings(dayNumber: number): {
  date: Date;
  formatted: string;           // "December 17"
  formattedWithYear: string;   // "December 17, 2025"
  formattedShort: string;      // "Dec 17"
  formattedWithWeekday: string; // "Wednesday, December 17"
  dayOfWeek: string;           // "Wednesday"
  monthName: string;           // "December"
  dayOfMonth: number;          // 17
  year: number;                // 2025
  dayNumber: number;           // 1 (internal, for URLs)
} {
  const date = dayNumberToDate(dayNumber);
  
  return {
    date,
    formatted: formatDateForDisplay(date),
    formattedWithYear: formatDateForDisplay(date, { includeYear: true }),
    formattedShort: formatDateForDisplay(date, { shortMonth: true }),
    formattedWithWeekday: formatDateForDisplay(date, { includeWeekday: true }),
    dayOfWeek: date.toLocaleDateString('en-US', { weekday: 'long' }),
    monthName: date.toLocaleDateString('en-US', { month: 'long' }),
    dayOfMonth: date.getDate(),
    year: date.getFullYear(),
    dayNumber // Keep for internal use (URLs, APIs)
  };
}

/**
 * Get footer text for emails (uses date not day number)
 * 
 * @param dayNumber - Internal day number
 * @returns Footer text like "December 17 • curiouskelly.com"
 */
export function getEmailFooterDate(dayNumber: number): string {
  const { formatted } = getLessonDateStrings(dayNumber);
  return `${formatted} • curiouskelly.com`;
}

/**
 * Check if a given day number is a special date
 * 
 * @param dayNumber - Internal day number
 * @returns Object with special date info
 */
export function getSpecialDateInfo(dayNumber: number): {
  isNewYearsEve: boolean;
  isNewYearsDay: boolean;
  isChristmas: boolean;
  isChristmasEve: boolean;
  isValentinesDay: boolean;
  isHalloween: boolean;
  isLaunchAnniversary: boolean;
  specialOccasion: string | null;
} {
  const date = dayNumberToDate(dayNumber);
  const month = date.getMonth();
  const day = date.getDate();
  
  const isNewYearsEve = month === 11 && day === 31;
  const isNewYearsDay = month === 0 && day === 1;
  const isChristmas = month === 11 && day === 25;
  const isChristmasEve = month === 11 && day === 24;
  const isValentinesDay = month === 1 && day === 14;
  const isHalloween = month === 9 && day === 31;
  const isLaunchAnniversary = month === 11 && day === 17 && dayNumber !== 1;
  
  let specialOccasion: string | null = null;
  if (isNewYearsDay) specialOccasion = 'New Year\'s Day';
  else if (isNewYearsEve) specialOccasion = 'New Year\'s Eve';
  else if (isChristmas) specialOccasion = 'Christmas';
  else if (isChristmasEve) specialOccasion = 'Christmas Eve';
  else if (isValentinesDay) specialOccasion = 'Valentine\'s Day';
  else if (isHalloween) specialOccasion = 'Halloween';
  else if (isLaunchAnniversary) specialOccasion = 'Kelly\'s Anniversary';
  
  return {
    isNewYearsEve,
    isNewYearsDay,
    isChristmas,
    isChristmasEve,
    isValentinesDay,
    isHalloween,
    isLaunchAnniversary,
    specialOccasion
  };
}

/**
 * Calculate days until a specific lesson day
 * 
 * @param targetDayNumber - Target day number
 * @returns Number of days until that lesson (negative if passed)
 */
export function daysUntilLesson(targetDayNumber: number): number {
  const today = getTodayLessonDay();
  return targetDayNumber - today;
}

/**
 * Get the lesson URL for a given day
 * 
 * @param dayNumber - Internal day number
 * @returns Full URL to the lesson
 */
export function getLessonUrl(dayNumber: number): string {
  return `https://curiouskelly.com/day/${dayNumber}`;
}

/**
 * Compute the next live class slot in the local environment timezone.
 *
 * Kelly is designed to start a new live class every N minutes
 * (default: 15). This helper rounds "now" up to the next interval
 * boundary and returns [start, end] as local Date objects.
 */
export function getNextLiveClassSlot(
  now: Date = new Date(),
  intervalMinutes = 15
): { start: Date; end: Date } {
  if (intervalMinutes <= 0) {
    throw new Error('intervalMinutes must be positive');
  }

  const start = new Date(now.getTime());
  start.setSeconds(0, 0);

  const minutes = start.getMinutes();
  const remainder = minutes % intervalMinutes;
  if (remainder !== 0) {
    start.setMinutes(minutes + (intervalMinutes - remainder));
  }

  const end = new Date(start.getTime() + intervalMinutes * 60_000);

  return { start, end };
}

// Export type for use in other modules
export interface LessonDateInfo {
  date: Date;
  formatted: string;
  formattedWithYear: string;
  formattedShort: string;
  formattedWithWeekday: string;
  dayOfWeek: string;
  monthName: string;
  dayOfMonth: number;
  year: number;
  dayNumber: number;
  url: string;
}

/**
 * Get complete lesson date info (combines all utilities)
 * 
 * @param dayNumber - Internal day number
 * @returns Complete lesson date information
 */
export function getCompleteLessonDateInfo(dayNumber: number): LessonDateInfo {
  const dateStrings = getLessonDateStrings(dayNumber);
  return {
    ...dateStrings,
    url: getLessonUrl(dayNumber)
  };
}



