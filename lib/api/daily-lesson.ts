/**
 * Sprint F: Daily Lesson API
 * Determines today's lesson and returns it
 * Falls back to database query if cache miss
 */
import { loadLesson, type LessonPayload } from './lesson-loader';

/**
 * Get the lesson day number for a given date
 * Uses day-of-year mod 365 (1-indexed)
 */
export function getLessonDayNumber(date: Date = new Date()): number {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  const dayOfYear = Math.floor(diff / oneDay);
  return ((dayOfYear - 1) % 365) + 1;
}

/**
 * Load today's daily lesson
 */
export async function getDailyLesson(
  ageGroup: string = 'adult',
  language: string = 'en'
): Promise<LessonPayload> {
  const dayNumber = getLessonDayNumber();
  return loadLesson(dayNumber, ageGroup, language);
}

/**
 * Load a specific day's lesson
 */
export async function getLessonForDay(
  dayNumber: number,
  ageGroup: string = 'adult',
  language: string = 'en'
): Promise<LessonPayload> {
  if (dayNumber < 1 || dayNumber > 365) {
    throw new Error(`Invalid day number: ${dayNumber}. Must be 1-365.`);
  }
  return loadLesson(dayNumber, ageGroup, language);
}

export type { LessonPayload };
