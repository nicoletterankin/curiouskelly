/**
 * ⏰🌍 LOCALIZATION, TIME & DATE LEARNINGS
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * This file documents the complete learnings, patterns, and best practices
 * for handling localization, time zones, and dates in the Curious Kelly system.
 * 
 * This is a REFERENCE DOCUMENT for future AI assistants and developers.
 * It captures the "why" behind our design decisions.
 * 
 * @see docs/architecture/TIME_AND_CALENDAR_LAW.md
 * @see KELLY_TIME_AUTHORITY.md
 * @see lib/lesson-dates.ts
 * 
 * Last Updated: December 24, 2025
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 1: CORE PHILOSOPHY
// ═══════════════════════════════════════════════════════════════════════════════

export const TIME_DATE_PHILOSOPHY = {
  tagline: "When Kelly says 9:00:00, class starts. Everywhere. For everyone.",
  
  coreBeliefs: {
    serverTimeIsTruth: "Client clocks can be wrong. Kelly's server clock (synced to NTP) is the single source of truth.",
    utcIsBackbone: "All internal timestamps use UTC. Local times are calculated from user timezone.",
    everySecondCounts: "Live classes start at :00 seconds. Countdowns are precise—'4:47', not 'about 5 minutes'.",
    topicsAreTimeless: "The same topic appears on the same calendar date every year. January 1 is always Day 1.",
    learnersSeeDates: "Users see real dates (December 25), never internal day numbers (Day 359).",
    predictability: "A learner in Tokyo and a learner in New York see the SAME topic for THEIR local date."
  },
  
  problemsWeSolve: [
    "Device clocks can be minutes or hours off",
    "Users travel across time zones",
    "Leap years create inconsistency in day-based systems",
    "Different regions use different date formats (MM/DD vs DD/MM)",
    "Daylight Saving Time shifts schedules unexpectedly",
    "Live classes need precision timing across the globe"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 2: THE CANONICAL CALENDAR SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

export const CANONICAL_CALENDAR = {
  overview: {
    calendarSystem: "Proleptic Gregorian",
    topicIndex: "day_number in range 1–365 (plus 366 for leap day)",
    anchor: "Day 1 = January 1, 2026 (topic 1), Day 365 = December 31 (topic 365)",
    invariant: "For any civil date (except Feb 29), the pair (month, day) ALWAYS maps to the SAME day_number"
  },
  
  mappingExamples: [
    { date: "January 1", dayNumber: 1, notes: "New Year's Day - always Day 1" },
    { date: "January 2", dayNumber: 2, notes: "Always Day 2" },
    { date: "February 14", dayNumber: 45, notes: "Valentine's Day - always Day 45" },
    { date: "December 25", dayNumber: 359, notes: "Christmas - always Day 359" },
    { date: "December 31", dayNumber: 365, notes: "New Year's Eve - always Day 365" }
  ],
  
  leapYearRules: {
    nonLeapYear: "topic_day = day_of_year (1–365)",
    leapYear: {
      jan1ToFeb28: "topic_day = day_of_year",
      feb29: "topic_day = 366 (special bonus lesson, shares no topic)",
      mar1ToDec31: "topic_day = day_of_year - 1"
    },
    consequences: [
      "March 1 is always topic 60",
      "December 31 is always topic 365",
      "February 29 gets its own special topic (366) in leap years"
    ]
  },
  
  workedExamples: {
    nonLeapYear2026: [
      { date: "2026-01-01", dayOfYear: 1, topicDay: 1 },
      { date: "2026-12-31", dayOfYear: 365, topicDay: 365 }
    ],
    leapYear2028: [
      { date: "2028-02-28", dayOfYear: 59, topicDay: 59 },
      { date: "2028-02-29", dayOfYear: 60, topicDay: 366, notes: "Leap day bonus" },
      { date: "2028-03-01", dayOfYear: 61, topicDay: 60 },
      { date: "2028-12-31", dayOfYear: 366, topicDay: 365 }
    ]
  },
  
  whyThisMatters: `
    This design ensures that topics are REUSABLE across years. When we create content
    for "Day 45" (Valentine's Day), that content works in 2026, 2027, 2028, forever.
    The spiral learning model means learners revisit the same topics annually, each
    time at a deeper level. The calendar anchoring makes this possible.
  `
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 3: TIMEZONE HANDLING PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

export const TIMEZONE_PATTERNS = {
  coreRule: "NEVER trust a device clock in isolation. Use server UTC + user timezone.",
  
  authoritativeInputs: {
    utcTimestamp: "From server / cron / Supabase (Date.now() on server)",
    userTimezone: "IANA string stored in Supabase users.timezone (e.g., 'America/Los_Angeles')"
  },
  
  canonicalComputation: [
    "1. Take (utcMillis, timeZone)",
    "2. Convert to user's LOCAL calendar date using Intl.DateTimeFormat with the specified timeZone",
    "3. Map that local date to day_number using dateToLessonDay() which encodes leap-year rules"
  ],
  
  keyFunctions: {
    getLessonDayForTimeZone: {
      signature: "getLessonDayForTimeZone(utcMillis?: number, timeZone?: string): number",
      description: "The PREFERRED entrypoint for anything user-facing or global",
      usage: "All user-facing flows should use this instead of raw `new Date()` math"
    }
  },
  
  timezoneDetection: {
    browser: "Intl.DateTimeFormat().resolvedOptions().timeZone",
    storage: "Store in Supabase users.timezone on account creation",
    fallback: "UTC if detection fails"
  },
  
  crossTimezoneExample: {
    scenario: "utcMillis corresponds to 2026-01-01T01:00:00Z",
    losAngeles: {
      timezone: "America/Los_Angeles (UTC-8 in winter)",
      localDate: "2025-12-31",
      topicDay: 365,
      notes: "Still December 31 locally"
    },
    tokyo: {
      timezone: "Asia/Tokyo (UTC+9)",
      localDate: "2026-01-01",
      topicDay: 1,
      notes: "Already January 1 locally"
    },
    result: "Kelly speaks about topic 365 in LA and topic 1 in Tokyo at the same UTC instant—both correct!"
  },
  
  commonMistakes: [
    "Using new Date() without an explicit timezone for user-facing mapping",
    "Computing day_number with ad-hoc math like ((now - launch) / 86400000) + 1",
    "Trusting Date.getDate() without considering timezone",
    "Forgetting that Date objects in JS are UTC internally"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 4: DATE FORMATTING STANDARDS
// ═══════════════════════════════════════════════════════════════════════════════

export const DATE_FORMATTING = {
  displayLocale: "en-US",
  
  coreRule: `
    Users ALWAYS see real dates ("December 25"), never internal day numbers ("Day 359").
    The day_number is internal only—for database, APIs, URLs, and scheduling.
  `,
  
  standardFormats: {
    formatted: { example: "December 17", usage: "Default display" },
    formattedWithYear: { example: "December 17, 2025", usage: "Emails, legal, historical" },
    formattedShort: { example: "Dec 17", usage: "Compact UI, mobile" },
    formattedWithWeekday: { example: "Wednesday, December 17", usage: "Full context" },
    dayOfWeek: { example: "Wednesday", usage: "Schedule displays" },
    monthName: { example: "December", usage: "Calendar headers" }
  },
  
  intlPatterns: {
    dateDisplay: `date.toLocaleDateString('en-US', { month: 'long', day: 'numeric' })`,
    withYear: `date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })`,
    shortMonth: `date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })`,
    withWeekday: `date.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })`,
    weekdayOnly: `date.toLocaleDateString('en-US', { weekday: 'long' })`,
    
    // For timezone-aware parsing (use en-CA for ISO-like output)
    timezoneParsing: `new Intl.DateTimeFormat('en-CA', { timeZone, year: 'numeric', month: '2-digit', day: '2-digit' })`
  },
  
  geoContextFormats: {
    dateFormatByCountry: "DD/MM/YYYY for most of world, MM/DD/YYYY for US/Canada",
    timeFormatByCountry: "24-hour for most of world, 12-hour for US/UK/Australia",
    implementation: "api/geo-context.ts provides dateFormat and use24Hour based on country"
  },
  
  emailFooterPattern: {
    format: "December 17 • curiouskelly.com",
    function: "getEmailFooterDate(dayNumber)",
    rule: "Always use formatted date, never day number"
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 5: LOCALIZATION ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════════════

export const LOCALIZATION_ARCHITECTURE = {
  goldenRule: "Languages are PRECOMPUTED in every DNA/content file (EN + ES/FR). NO runtime language generation.",
  
  supportedLanguages: {
    current: ["en", "es", "fr"],
    planned: ["pt", "de", "ja", "zh", "ar", "hi"],
    model: "eleven_multilingual_v2 for TTS"
  },
  
  contentStructure: {
    lessonAtoms: "lesson_atoms.content JSONB contains multilingual scripts",
    lessonShards: "~6,570 age/language variants",
    precomputed: "All content has EN + ES/FR embedded in database at generation time"
  },
  
  whyPrecompute: [
    "Runtime generation is slow and expensive",
    "Quality control requires human review before publish",
    "Caching becomes trivial when content is static",
    "Consistency across devices and sessions",
    "ElevenLabs API costs are controlled by batch generation"
  ],
  
  languageDetection: {
    primary: "User preference stored in database",
    fallback: "api/geo-context.ts provides suggestedLanguage from country code",
    browserDetection: "navigator.language as last resort",
    countryMapping: "COUNTRY_TO_LANGUAGE map in geo-context.ts"
  },
  
  neverDo: [
    "Generate language content at runtime",
    "Use browser TTS for any language",
    "Store language preference only in localStorage (must sync to server)",
    "Assume English as default without checking user preferences"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 6: GEO-CONTEXT API
// ═══════════════════════════════════════════════════════════════════════════════

export const GEO_CONTEXT = {
  endpoint: "/api/geo-context",
  
  providedData: {
    location: {
      country: "ISO 3166-1 alpha-2 code (e.g., 'US')",
      countryName: "Full country name",
      region: "State/province",
      city: "City name"
    },
    time: {
      timezone: "IANA timezone string",
      localTime: "Formatted local time",
      localDate: "Formatted local date",
      hour: "Current hour (0-23)",
      minute: "Current minute",
      timeOfDay: "'morning' | 'afternoon' | 'evening' | 'night'"
    },
    calendar: {
      dayOfWeek: "Full weekday name",
      dayOfWeekShort: "3-letter weekday",
      dayOfWeekNumber: "0-6 (Sunday = 0)",
      dayOfMonth: "1-31",
      dayOfYear: "1-366",
      weekOfYear: "1-52",
      month: "Full month name",
      monthNumber: "0-11",
      year: "Full year number",
      isWeekend: "Boolean"
    },
    season: {
      season: "'spring' | 'summer' | 'autumn' | 'winter'",
      hemisphere: "'northern' | 'southern'",
      note: "Season inverts for southern hemisphere"
    },
    preferences: {
      suggestedLanguage: "Based on country",
      dateFormat: "'DD/MM/YYYY' or 'MM/DD/YYYY' based on country",
      use24Hour: "Boolean based on country"
    }
  },
  
  usagePatterns: {
    greetingByTimeOfDay: "Good morning/afternoon/evening based on hour",
    seasonalContent: "Summer vs winter framing based on hemisphere",
    formatPreferences: "Respect local date/time format conventions"
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 7: SERVER TIME SYNCHRONIZATION
// ═══════════════════════════════════════════════════════════════════════════════

export const TIME_SYNC = {
  endpoint: "/api/time",
  
  response: {
    utc: "Unix timestamp in milliseconds",
    iso: "ISO 8601 string",
    unix: "Unix timestamp in seconds"
  },
  
  clientSyncAlgorithm: {
    description: "NTP-like algorithm to account for network latency",
    steps: [
      "1. Record client time t1 before request",
      "2. Fetch server time",
      "3. Record client time t4 after response",
      "4. Calculate round-trip time (RTT) = t4 - t1",
      "5. Estimate one-way latency = RTT / 2",
      "6. Calculate offset = serverTime + latency - t4"
    ],
    usage: "Use offset to adjust all client-side time displays"
  },
  
  syncInterval: "60 seconds (configurable)",
  
  fallback: "If sync fails, use client time with warning logged"
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 8: LIVE CLASS SCHEDULING
// ═══════════════════════════════════════════════════════════════════════════════

export const LIVE_CLASS_SCHEDULING = {
  intervalMinutes: 15,
  duration: 15,
  
  schedulePattern: {
    description: "Classes start every 15 minutes, on the quarter hour",
    examples: ["9:00", "9:15", "9:30", "9:45", "10:00"],
    rule: "Round current time UP to next 15-minute boundary"
  },
  
  function: {
    name: "getNextLiveClassSlot",
    signature: "(now?: Date, intervalMinutes?: number) => { start: Date, end: Date }",
    description: "Returns next class start/end times in local timezone"
  },
  
  displayFormat: {
    countdown: "4h 23m 15s (updates every second)",
    liveNow: "'LIVE NOW' when class is in session",
    nextClass: "'Next: Morning class at 9:00 AM'"
  },
  
  classLabels: {
    6: "Early Birds",
    9: "Morning",
    12: "Lunch",
    18: "Evening",
    21: "Night Owls"
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 9: CALENDAR INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

export const CALENDAR_INTEGRATION = {
  icsFormat: {
    version: "2.0",
    prodId: "-//Curious Kelly//Daily Lessons//EN",
    calScale: "GREGORIAN"
  },
  
  supportedCalendars: {
    google: "calendar.google.com deep link",
    apple: "webcal:// protocol for .ics subscription",
    outlook: "outlook.live.com deep link",
    yahoo: "calendar.yahoo.com deep link",
    ics: "Universal .ics file download"
  },
  
  eventTypes: {
    singleLesson: "One-time event for today's lesson",
    dailyReminder: "Recurring daily event at user's preferred time",
    liveClassSchedule: "Multiple recurring events for all class times",
    subscriptionFeed: "Auto-updating calendar with all 365 lessons"
  },
  
  icsDateFormat: {
    pattern: "YYYYMMDDTHHMMSSZ",
    implementation: "date.toISOString().replace(/[-:]/g, '').replace(/\\.\\d{3}/, '')"
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 10: SPECIAL DATES & OCCASIONS
// ═══════════════════════════════════════════════════════════════════════════════

export const SPECIAL_DATES = {
  tracked: [
    { name: "New Year's Day", dayNumber: 1, month: 0, day: 1 },
    { name: "Valentine's Day", dayNumber: 45, month: 1, day: 14 },
    { name: "Halloween", dayNumber: 304, month: 9, day: 31 },
    { name: "Christmas Eve", dayNumber: 358, month: 11, day: 24 },
    { name: "Christmas", dayNumber: 359, month: 11, day: 25 },
    { name: "New Year's Eve", dayNumber: 365, month: 11, day: 31 },
    { name: "Kelly's Anniversary", dayNumber: 351, month: 11, day: 17, notes: "Launch day" }
  ],
  
  function: {
    name: "getSpecialDateInfo",
    returns: {
      isNewYearsEve: "boolean",
      isNewYearsDay: "boolean",
      isChristmas: "boolean",
      isChristmasEve: "boolean",
      isValentinesDay: "boolean",
      isHalloween: "boolean",
      isLaunchAnniversary: "boolean",
      specialOccasion: "string | null"
    }
  },
  
  usage: "Special theming, greetings, and content variations for holidays"
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 11: ANTI-PATTERNS & GUARDRAILS
// ═══════════════════════════════════════════════════════════════════════════════

export const ANTI_PATTERNS = {
  neverDo: [
    "Compute day_number with ad-hoc math (e.g., ((now - launch) / 86400000) + 1)",
    "Use new Date() without an explicit timezone for user-facing mapping",
    "Introduce a 366th topic for Feb 29 (it has its own special topic)",
    "Hard-code 'Day 1 = December 17, 2025' anywhere (Day 1 = January 1, 2026)",
    "Trust device clock without server sync for live features",
    "Generate language content at runtime",
    "Use browser TTS",
    "Display day numbers to users (always show dates)"
  ],
  
  alwaysDo: [
    "Use lib/lesson-dates.ts as the SINGLE SOURCE OF TRUTH",
    "Pass (utcMillis, timeZone) into getLessonDayForTimeZone wherever possible",
    "Log both utcMillis and timeZone when debugging calendar issues",
    "Precompute all language content",
    "Use Intl.DateTimeFormat for all date/time formatting",
    "Store user timezone in database, not just localStorage"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 12: KEY FILES REFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

export const KEY_FILES = {
  implementation: {
    "lib/lesson-dates.ts": "Core date/day conversions, formatting, leap year logic",
    "public/js/kelly-time.js": "Client-side time sync and display",
    "api/time.ts": "Server time endpoint",
    "api/geo-context.ts": "Geographic and timezone context"
  },
  
  documentation: {
    "docs/architecture/TIME_AND_CALENDAR_LAW.md": "Canonical calendar rules",
    "KELLY_TIME_AUTHORITY.md": "Time sync, clock display, calendar integration"
  },
  
  usage: {
    "api/cron/daily-push-notifications.ts": "Uses getLessonDayForTimeZone for user notifications",
    "api/cron/daily-lesson.ts": "Daily email with correct local date",
    "public/learn.html": "Lesson player date display"
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 13: TESTING REQUIREMENTS
// ═══════════════════════════════════════════════════════════════════════════════

export const TESTING_REQUIREMENTS = {
  unitTests: {
    location: "tests/unit/lesson-dates-law.test.ts",
    coverage: [
      "Non-leap vs leap year behavior (especially Feb 28/29 and Mar 1)",
      "Stability of mappings for key dates across multiple years",
      "Correct behavior of getLessonDayForTimeZone in multiple time zones",
      "Correct quarter-hour rounding for getNextLiveClassSlot"
    ]
  },
  
  manualVerification: [
    "Time sync: curl /api/time and verify response",
    "Clock updates every second (inspect element in browser)",
    "Countdown accuracy: wait until :00:00, confirm 'LIVE NOW' appears",
    "ICS generation: verify valid calendar file",
    "Multi-timezone: test with different browser timezone settings"
  ],
  
  invariant: "Any change to lib/lesson-dates.ts MUST keep tests passing"
};

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 14: INTEGRATION CHECKLIST
// ═══════════════════════════════════════════════════════════════════════════════

export const INTEGRATION_CHECKLIST = {
  supabase: [
    "core_lessons.day_number is canonical topic index (1–365)",
    "Lesson fetches use same day_number computed by lesson-dates.ts",
    "users.timezone stores IANA timezone string"
  ],
  
  notifications: [
    "Fetch user timezone from Supabase",
    "Call getLessonDayForTimeZone(Date.now(), user.timezone)",
    "Use day_number for selecting core_lessons record",
    "Use getLessonDateStrings for subject lines and footers"
  ],
  
  lessonPlayer: [
    "Show real local date (e.g., 'Wednesday, January 1')",
    "Keep internal URLs aligned to day_number",
    "For live class UI, use getNextLiveClassSlot()"
  ],
  
  multiArchetype: [
    "All archetypes share SAME day_number",
    "Language and tone variants share SAME day_number",
    "Cache keys: (day_number, archetype, language, tone)"
  ]
};

// ═══════════════════════════════════════════════════════════════════════════════
// QUICK REFERENCE: COPY-PASTE PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

export const CODE_PATTERNS = {
  getTodaysLesson: `
// Get today's lesson for a user
import { getLessonDayForTimeZone } from '@/lib/lesson-dates';

const userTimezone = user.timezone || 'UTC';
const dayNumber = getLessonDayForTimeZone(Date.now(), userTimezone);
// Use dayNumber to fetch from core_lessons
`,

  formatDateForUser: `
// Format a date for display
import { getLessonDateStrings } from '@/lib/lesson-dates';

const { formatted, formattedWithWeekday, dayOfWeek } = getLessonDateStrings(dayNumber);
// formatted: "December 25"
// formattedWithWeekday: "Thursday, December 25"
`,

  detectTimezone: `
// Detect user's timezone in browser
const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
// Returns: "America/New_York"
`,

  formatTimeInTimezone: `
// Format time for specific timezone
const formatted = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Los_Angeles',
  hour: 'numeric',
  minute: '2-digit',
  hour12: true
}).format(new Date());
// Returns: "9:15 AM"
`,

  parseTimezoneAwareDate: `
// Get date parts in a specific timezone
const formatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: userTimezone,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit'
});
const parts = formatter.formatToParts(new Date());
// Parse parts to get year, month, day in user's local time
`,

  liveClassNextSlot: `
// Get next live class time
import { getNextLiveClassSlot } from '@/lib/lesson-dates';

const { start, end } = getNextLiveClassSlot();
// start: Date for next class start (on quarter hour)
// end: Date for class end (15 min later)
`
};

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORT SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

export default {
  philosophy: TIME_DATE_PHILOSOPHY,
  calendar: CANONICAL_CALENDAR,
  timezones: TIMEZONE_PATTERNS,
  formatting: DATE_FORMATTING,
  localization: LOCALIZATION_ARCHITECTURE,
  geoContext: GEO_CONTEXT,
  timeSync: TIME_SYNC,
  liveClasses: LIVE_CLASS_SCHEDULING,
  calendarIntegration: CALENDAR_INTEGRATION,
  specialDates: SPECIAL_DATES,
  antiPatterns: ANTI_PATTERNS,
  keyFiles: KEY_FILES,
  testing: TESTING_REQUIREMENTS,
  integration: INTEGRATION_CHECKLIST,
  codePatterns: CODE_PATTERNS
};



