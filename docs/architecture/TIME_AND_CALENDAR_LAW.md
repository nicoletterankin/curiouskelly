# ⏰ Curious Kelly Time & Calendar Law

> **Goal:** One canonical “topic of the day” that is stable across years, leap years, time zones, archetypes, languages, and devices.

---

## 1. Canonical Topic Calendar

- **Calendar system**: Proleptic Gregorian.
- **Topic index**: `day_number` in the range **1–365**.
- **Anchor**: **Day 1 = January 1, 2026** (topic 1), **Day 365 = December 31** (topic 365).
- **Invariant**: For any civil date (except Feb 29), the pair `(month, day)` always maps to the **same `day_number`**:
  - January 1 → 1  
  - January 2 → 2  
  - ...  
  - December 31 → 365  

Lessons, notifications, analytics, and UI all speak in terms of `day_number`. Assets and copy for a given `day_number` are reusable across years.

---

## 2. Leap Year Rules (Non‑Negotiable)

Leap years introduce a 366th calendar day (Feb 29). We **do not** introduce a 366th topic.

- **Non‑leap years**:  
  - `topic_day = day_of_year` (1–365)

- **Leap years**:
  - Jan 1 – Feb 28 → `topic_day = day_of_year`
  - Feb 29 → `topic_day = 60` (shares topic with March 1)
  - Mar 1 – Dec 31 → `topic_day = day_of_year - 1`

Consequences:
- March 1 is always topic 60.
- December 31 is always topic 365.
- February 29, when present, **reuses** the March 1 topic. It never gets its own topic ID.

This is encoded in `lib/lesson-dates.ts` and must not be re‑implemented ad‑hoc anywhere else.

---

## 3. Time Zones & Device Clocks

We never trust a device clock in isolation.

**Authoritative inputs:**
- **UTC timestamp** (from server / cron / Supabase).
- **User timezone** as an IANA string (e.g. `America/Los_Angeles`), stored in Supabase `users.timezone`.

**Canonical computation:**
1. Take `(utcMillis, timeZone)`.
2. Convert to the user’s **local calendar date** using `Intl.DateTimeFormat` with the specified `timeZone`.
3. Map that local date to `day_number` using `dateToLessonDay()` (which encodes the leap‑year rules).

In code, this is done via:

- `getLessonDayForTimeZone(utcMillis?: number, timeZone?: string): number`

All user‑facing flows should use this function instead of raw `new Date()` math.

---

## 4. Library API Surface (`lib/lesson-dates.ts`)

**Core conversions**
- `dateToLessonDay(date: Date): number`  
  - Input: local `Date` (already in correct timezone).  
  - Output: canonical `day_number` (1–365), with leap‑year compression applied.

- `dayNumberToDate(dayNumber: number, year?: number): Date`  
  - Input: canonical `day_number` (1–365) and a concrete calendar `year`.  
  - Output: actual `Date` in that year. On leap years, topics ≥60 are shifted forward by one day to insert Feb 29.

**Timezone‑aware helper**
- `getLessonDayForTimeZone(utcMillis?: number, timeZone?: string): number`  
  - Preferred entrypoint for anything user‑facing or global.  
  - Uses `(utcMillis, timeZone)` → local date → `dateToLessonDay`.

**Formatting helpers**
- `formatDateForDisplay(date, options)` → `"January 1"`, `"Wed, Jan 1, 2026"`, etc.
- `getLessonDateStrings(dayNumber)` → `formatted`, `formattedWithYear`, `dayOfWeek`, `monthName`, etc.
- `getEmailFooterDate(dayNumber)` → `"January 1 • curiouskelly.com"`.
- `getLessonUrl(dayNumber)` → `https://curiouskelly.com/day/{day_number}`.

**Live class scheduling**
- `getNextLiveClassSlot(now?: Date, intervalMinutes = 15)`  
  - Round “now” up to the next **15‑minute** boundary and return `{ start, end }` in the local environment timezone.
  - Used to show: “Kelly’s next live class starts at 7:15 pm your time.”

---

## 5. Integration Points (Must Use the Law)

### 5.1 Supabase & Lesson Content

- `core_lessons.day_number` is the canonical topic index (1–365).
- Lesson fetches must use **the same `day_number`** computed by `lesson-dates.ts`:
  - Web / app lesson player.
  - Cron jobs (daily email, daily push, birthday fusion).
  - Analytics and streak tracking.

### 5.2 Notification System

- Cron jobs (`api/cron/daily-lesson.ts`, `api/cron/daily-push-notifications.ts`) should:
  1. Fetch the user’s timezone from Supabase.
  2. Call `getLessonDayForTimeZone(Date.now(), user.timezone)`.
  3. Use that `day_number` when:
     - Selecting today’s `core_lessons` record.
     - Computing subject lines.
     - Rendering email footer dates (via `getLessonDateStrings` / `getEmailFooterDate`).

### 5.3 Lesson Player & Live Experiences

- The lesson player should:
  - Show the **real local date** (e.g., “Wednesday, January 1”) using `getLessonDateStrings`.
  - Keep internal URLs and API calls aligned to `day_number`.
  - For live class UI, call `getNextLiveClassSlot()` and render:
    - Local time (start/end).
    - Local date and weekday.
    - Topic title for `day_number`.

### 5.4 Multi‑Archetype / Multi‑Language

- All archetypes, languages, and tones share the **same `day_number`**:
  - A Survivor archetype in Spanish and a Scientist archetype in English looking at January 1 should both be on **topic 1**, just expressed differently.
- This makes it safe to:
  - Cache assets keyed by `(day_number, archetype, language, tone)`.
  - Switch archetypes without changing “what day it is.”

---

## 6. Worked Examples

### 6.1 Non‑Leap Year (2026)

- `2026‑01‑01` (any timezone where it is Jan 1 locally):
  - `day_of_year = 1`
  - `topic_day = 1`

- `2026‑12‑31`:
  - `day_of_year = 365`
  - `topic_day = 365`

### 6.2 Leap Year (2028)

- `2028‑02‑28`:
  - `day_of_year = 59`
  - `topic_day = 59`

- `2028‑02‑29`:
  - `day_of_year = 60`  
  - **Leap rule applies** → `topic_day = 60` (shares March 1 topic)

- `2028‑03‑01`:
  - `day_of_year = 61`
  - `topic_day = 61 − 1 = 60`

- `2028‑12‑31`:
  - `day_of_year = 366`
  - `topic_day = 366 − 1 = 365`

### 6.3 Time Zone Example

Assume:
- `utcMillis` corresponds to `2026‑01‑01T01:00:00Z`.

- For `America/Los_Angeles` (UTC‑8 in winter):
  - Local date is `2025‑12‑31`.
  - `topic_day = 365` (previous year’s final topic).

- For `Asia/Tokyo` (UTC+9):
  - Local date is `2026‑01‑01`.
  - `topic_day = 1`.

Kelly therefore speaks about **topic 365** in Los Angeles and **topic 1** in Tokyo at the same instant in UTC, both correct for the local date.

---

## 7. Guardrails & Anti‑Patterns

- **Do not**:
  - Compute `day_number` with ad‑hoc math (e.g., `((now - launch) / 86400000) + 1`).
  - Use `new Date()` without an explicit timezone for any user‑facing mapping.
  - Introduce a 366th topic for Feb 29.
  - Hard‑code “Day 1 = December 17, 2025” anywhere going forward.

- **Do**:
  - Use `lib/lesson-dates.ts` as the **single source of truth**.
  - Pass `(utcMillis, timeZone)` into `getLessonDayForTimeZone` wherever possible.
  - Log both `utcMillis` and `timeZone` when debugging calendar issues.

---

## 8. Testing Notes

Unit tests (in `tests/unit/lesson-dates-law.test.ts`) validate:
- Non‑leap vs leap behavior (especially Feb 28/29 and Mar 1).
- Stability of mappings for key dates across multiple years.
- Correct behavior of `getLessonDayForTimeZone` in multiple time zones.
- Correct quarter‑hour rounding for `getNextLiveClassSlot`.

Any change to `lib/lesson-dates.ts` must keep these tests passing.
























