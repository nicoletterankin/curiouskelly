/**
 * KellyLesson — shared lesson identity helpers (browser).
 *
 * Purpose:
 * - Centralize "what lesson is today" and "how do we label it" without ever showing "Day X".
 *
 * Depends on `window.KellyTime` (load kelly-time.js first).
 */
(() => {
  function requireKellyTime() {
    if (!window.KellyTime) {
      throw new Error('KellyLesson requires window.KellyTime. Load /js/kelly-time.js first.');
    }
    return window.KellyTime;
  }

  function getSupabase() {
    // Prefer the shared singleton to avoid multiple GoTrueClient instances.
    if (typeof window.getSupabase === 'function') return window.getSupabase();
    if (window.supabaseClient) return window.supabaseClient;
    return null;
  }

  function getTodayLessonKey(options = {}) {
    const KellyTime = requireKellyTime();
    const { utcMillis = Date.now(), timeZone = KellyTime.getUserTimeZone() } = options;
    const dayNumber = KellyTime.getLessonDayForTimeZone(utcMillis, timeZone);
    const displayDate = KellyTime.formatDayNumber(dayNumber, { utcMillis, timeZone, includeYear: true });
    return { dayNumber, displayDate, timeZone, utcMillis };
  }

  function getLessonDisplayDate(dayNumber, options = {}) {
    const KellyTime = requireKellyTime();
    const { utcMillis = Date.now(), timeZone = KellyTime.getUserTimeZone(), includeWeekday = false } = options;
    return KellyTime.formatDayNumber(dayNumber, { utcMillis, timeZone, includeYear: true, includeWeekday });
  }

  const KellyLesson = {
    getTodayLessonKey,
    getLessonDisplayDate,
    getSupabase,
  };

  window.KellyLesson = KellyLesson;
})();

