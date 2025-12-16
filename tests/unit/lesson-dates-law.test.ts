import { describe, it, expect } from 'vitest';

import {
  dateToLessonDay,
  dayNumberToDate,
  getLessonDayForTimeZone,
  getNextLiveClassSlot,
} from '../../lib/lesson-dates';

describe('Curious Kelly Time & Calendar Law', () => {
  it('maps non-leap year dates to 1–365', () => {
    const jan1 = new Date(2026, 0, 1);
    const dec31 = new Date(2026, 11, 31);

    expect(dateToLessonDay(jan1)).toBe(1);
    expect(dateToLessonDay(dec31)).toBe(365);
  });

  it('handles leap-year rules correctly (Feb 29 is a special bonus lesson)', () => {
    const feb28 = new Date(2028, 1, 28); // 2028 is a leap year
    const feb29 = new Date(2028, 1, 29);
    const mar1 = new Date(2028, 2, 1);
    const dec31 = new Date(2028, 11, 31);

    expect(dateToLessonDay(feb28)).toBe(59);
    expect(dateToLessonDay(feb29)).toBe(366);
    expect(dateToLessonDay(mar1)).toBe(60);
    expect(dateToLessonDay(dec31)).toBe(365);
  });

  it('round-trips dayNumberToDate in leap and non-leap years', () => {
    // Non-leap year
    const d1_2026 = dayNumberToDate(1, 2026);
    const d365_2026 = dayNumberToDate(365, 2026);
    expect(d1_2026.getFullYear()).toBe(2026);
    expect(d1_2026.getMonth()).toBe(0);
    expect(d1_2026.getDate()).toBe(1);
    expect(d365_2026.getFullYear()).toBe(2026);
    expect(d365_2026.getMonth()).toBe(11);
    expect(d365_2026.getDate()).toBe(31);

    // Leap year
    const d59_2028 = dayNumberToDate(59, 2028);
    const d60_2028 = dayNumberToDate(60, 2028);
    const d365_2028 = dayNumberToDate(365, 2028);

    expect(d59_2028.getFullYear()).toBe(2028);
    expect(d59_2028.getMonth()).toBe(1);
    expect(d59_2028.getDate()).toBe(28);

    // 60th topic day should land on March 1 in leap years
    expect(d60_2028.getFullYear()).toBe(2028);
    expect(d60_2028.getMonth()).toBe(2);
    expect(d60_2028.getDate()).toBe(1);

    expect(d365_2028.getFullYear()).toBe(2028);
    expect(d365_2028.getMonth()).toBe(11);
    expect(d365_2028.getDate()).toBe(31);
  });

  it('computes lesson day per timezone from a single UTC instant', () => {
    // 2026-01-01T01:00:00Z
    const utc = Date.UTC(2026, 0, 1, 1, 0, 0, 0);

    const laDay = getLessonDayForTimeZone(utc, 'America/Los_Angeles');
    const nyDay = getLessonDayForTimeZone(utc, 'America/New_York');
    const tokyoDay = getLessonDayForTimeZone(utc, 'Asia/Tokyo');

    // In LA, this is still Dec 31, 2025 → topic 365
    expect(laDay).toBe(365);
    // In New York, this is Jan 1, 2026 → topic 1
    expect(nyDay).toBe(1);
    // In Tokyo, this is already Jan 1, 2026 → topic 1
    expect(tokyoDay).toBe(1);
  });

  it('rounds to the next live class slot at 15-minute intervals', () => {
    const base = new Date(2026, 0, 1, 9, 7, 30, 500); // 9:07:30.500
    const { start, end } = getNextLiveClassSlot(base, 15);

    expect(start.getHours()).toBe(9);
    expect(start.getMinutes()).toBe(15);
    expect(start.getSeconds()).toBe(0);
    expect(start.getMilliseconds()).toBe(0);

    expect(end.getHours()).toBe(9);
    expect(end.getMinutes()).toBe(30);
  });
});









