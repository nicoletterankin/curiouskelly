/**
 * KellyCalendar — shared calendar utilities for Kelly OS (browser).
 *
 * Depends on `window.KellyTime` (load kelly-time.js first).
 */
(() => {
  function requireKellyTime() {
    if (!window.KellyTime) {
      throw new Error('KellyCalendar requires window.KellyTime. Load /js/kelly-time.js first.');
    }
    return window.KellyTime;
  }

  /**
   * Produce a week label like: "December 8–December 14, 2025"
   * using canonical dayNumbers (1–365) within the given year.
   */
  function formatWeekRangeLabel(startDayNumber, year, options = {}) {
    const KellyTime = requireKellyTime();
    const { utcMillis = Date.now(), timeZone = KellyTime.getUserTimeZone() } = options;
    const y = year ?? KellyTime.getLocalDateParts(utcMillis, timeZone).year;

    const startDate = KellyTime.dayNumberToDate(startDayNumber, y);
    const endDate = KellyTime.dayNumberToDate(Math.min(startDayNumber + 6, 365), y);

    const sameMonth = startDate.getMonth() === endDate.getMonth();
    if (sameMonth) {
      const monthName = startDate.toLocaleDateString('en-US', { month: 'long' });
      return `${monthName} ${startDate.getDate()}–${endDate.getDate()}, ${y}`;
    }

    const startStr = KellyTime.formatDate(startDate, { includeYear: false, shortMonth: false });
    const endStr = KellyTime.formatDate(endDate, { includeYear: false, shortMonth: false });
    return `${startStr}–${endStr}, ${y}`;
  }

  /**
   * Build month metadata (month name + days + mapping to canonical dayNumbers).
   * Returns an array of 12 entries for the given year.
   */
  function getYearMonths(year, options = {}) {
    const KellyTime = requireKellyTime();
    const { utcMillis = Date.now(), timeZone = KellyTime.getUserTimeZone() } = options;
    const y = year ?? KellyTime.getLocalDateParts(utcMillis, timeZone).year;

    const months = [];
    let dayNumber = 1;
    for (let m = 0; m < 12; m++) {
      const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m];
      months.push({
        year: y,
        monthIndex: m,
        monthName: new Date(y, m, 1).toLocaleDateString('en-US', { month: 'long' }),
        daysInMonth,
        dayNumberStart: dayNumber,
        dayNumberEnd: dayNumber + daysInMonth - 1,
      });
      dayNumber += daysInMonth;
    }
    return months;
  }

  const KellyCalendar = {
    formatWeekRangeLabel,
    getYearMonths,
  };

  window.KellyCalendar = KellyCalendar;
})();

