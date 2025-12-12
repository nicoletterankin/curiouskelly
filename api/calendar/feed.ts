/**
 * /api/calendar/feed.ts - Kelly Calendar Feed
 * 
 * Subscribable ICS calendar feed for all Kelly lessons.
 * Users can add this to Google Calendar, Apple Calendar, Outlook, etc.
 * 
 * Usage:
 * - Subscribe URL: https://curiouskelly.com/api/calendar/feed
 * - Google Calendar: webcal://curiouskelly.com/api/calendar/feed
 * - Apple Calendar: webcal://curiouskelly.com/api/calendar/feed
 */

import { createClient } from '@supabase/supabase-js';

export const config = {
  runtime: 'edge',
};

// Live class schedule (local hours)
const LIVE_CLASS_HOURS = [6, 9, 12, 18, 21];
const LIVE_CLASS_LABELS: Record<number, string> = {
  6: 'Early Birds',
  9: 'Morning',
  12: 'Lunch',
  18: 'Evening',
  21: 'Night Owls',
};

function formatICSDate(date: Date): string {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
}

function escapeICS(text: string): string {
  return text
    .replace(/\\/g, '\\\\')
    .replace(/,/g, '\\,')
    .replace(/;/g, '\\;')
    .replace(/\n/g, '\\n');
}

function generateUID(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}@curiouskelly.com`;
}

export default async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const feedType = url.searchParams.get('type') || 'lessons';
  
  const now = new Date();
  const year = now.getFullYear();
  
  // ICS header
  const icsLines: string[] = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//Curious Kelly//Kelly Time Authority//EN',
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    'X-WR-CALNAME:Curious Kelly',
    'X-WR-CALDESC:Daily 5-minute lessons from Kelly',
    'X-WR-TIMEZONE:UTC',
  ];
  
  if (feedType === 'live') {
    // Generate live class schedule
    LIVE_CLASS_HOURS.forEach((hour) => {
      const label = LIVE_CLASS_LABELS[hour];
      const start = new Date(year, now.getMonth(), now.getDate(), hour, 0, 0);
      const end = new Date(start);
      end.setMinutes(15);
      
      icsLines.push(
        'BEGIN:VEVENT',
        `UID:${generateUID(`live-${hour}`)}`,
        `DTSTAMP:${formatICSDate(now)}`,
        `DTSTART:${formatICSDate(start)}`,
        `DTEND:${formatICSDate(end)}`,
        `SUMMARY:Kelly LIVE: ${label}`,
        `DESCRIPTION:${escapeICS(`Join the ${label} live class with Kelly and learners worldwide.\\n\\nJoin: https://curiouskelly.com/live.html`)}`,
        'URL:https://curiouskelly.com/live.html',
        'RRULE:FREQ=DAILY',
        `CATEGORIES:Live Class`,
        'END:VEVENT'
      );
    });
  } else {
    // Generate daily lesson events
    // For now, generate 30 days of lessons
    for (let dayOffset = 0; dayOffset < 30; dayOffset++) {
      const lessonDate = new Date(year, now.getMonth(), now.getDate() + dayOffset, 9, 0, 0);
      const endDate = new Date(lessonDate);
      endDate.setMinutes(5); // 5-minute lesson
      
      const dayNumber = Math.floor((lessonDate.getTime() - new Date(year, 0, 0).getTime()) / (1000 * 60 * 60 * 24));
      
      icsLines.push(
        'BEGIN:VEVENT',
        `UID:${generateUID(`lesson-${dayNumber}`)}`,
        `DTSTAMP:${formatICSDate(now)}`,
        `DTSTART:${formatICSDate(lessonDate)}`,
        `DTEND:${formatICSDate(endDate)}`,
        `SUMMARY:Kelly: Daily Lesson`,
        `DESCRIPTION:${escapeICS(`Your daily 5-minute lesson from Curious Kelly.\\n\\nOpen: https://curiouskelly.com/learn.html`)}`,
        'URL:https://curiouskelly.com/learn.html',
        `CATEGORIES:Daily Lesson`,
        'END:VEVENT'
      );
    }
  }
  
  icsLines.push('END:VCALENDAR');
  
  const icsContent = icsLines.join('\r\n');
  
  return new Response(icsContent, {
    status: 200,
    headers: {
      'Content-Type': 'text/calendar; charset=utf-8',
      'Content-Disposition': `attachment; filename="kelly-${feedType}.ics"`,
      // Cache for 1 hour - lessons don't change that often
      'Cache-Control': 'public, max-age=3600',
      'Access-Control-Allow-Origin': '*',
    },
  });
}

