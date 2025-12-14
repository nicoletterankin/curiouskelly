/**
 * Lessons API - Vercel Edge Function
 * 
 * Primary: serves lessons from Supabase (service role) so unauthenticated
 * users can load lesson metadata and build the 365-day calendar without auth.
 *
 * Fallback: returns a small static set (always available).
 * 
 * GET /api/lessons/:dayNumber
 */

// Simple static lesson data for fallback
const STATIC_LESSONS: Record<number, any> = {
  1: { topic: 'The Sun', universal_truth: 'Our star gives life to everything on Earth.' },
  2: { topic: 'Why the Sky is Blue', universal_truth: 'Light bends and scatters to paint our world.' },
  3: { topic: 'How Seeds Grow', universal_truth: 'Every giant oak began as a tiny seed.' },
  4: { topic: 'The Water Cycle', universal_truth: 'Water is never created or destroyed—only transformed.' },
  5: { topic: 'Why We Sleep', universal_truth: 'Sleep is when your brain organizes everything you learned.' },
  6: { topic: 'How Birds Fly', universal_truth: 'Nature solved flight millions of years before humans.' },
  7: { topic: 'The Moon', universal_truth: 'The Moon has watched over Earth for 4.5 billion years.' },
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Cache-Control': 'public, max-age=300',
      'Content-Type': 'application/json; charset=utf-8',
    },
  });
}

export default async function handler(req: Request): Promise<Response> {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Cache-Control': 'public, max-age=300',
      },
    });
  }

  // Edge Runtime: parse query params from URL
  const url = new URL(req.url);
  const dayParam = url.searchParams.get('dayNumber') || url.searchParams.get('day') || '1';
  const startParam = url.searchParams.get('startDay') || url.searchParams.get('start') || null;
  const endParam = url.searchParams.get('endDay') || url.searchParams.get('end') || null;
  const day = Number.parseInt(dayParam, 10);

  // ---------------------------------------------------------------------------
  // Range endpoint: /api/lessons?startDay=1&endDay=365
  // ---------------------------------------------------------------------------
  if (startParam && endParam) {
    const startDay = Number.parseInt(startParam, 10);
    const endDay = Number.parseInt(endParam, 10);

    if (
      Number.isNaN(startDay) ||
      Number.isNaN(endDay) ||
      startDay < 1 ||
      endDay > 365 ||
      startDay > endDay
    ) {
      return jsonResponse(
        { error: 'Invalid range', message: 'startDay/endDay must be 1..365 and startDay <= endDay' },
        400
      );
    }

    try {
      // Lazy import to keep Edge bundle small and avoid hard dependency when env vars missing
      const { getSupabaseAdmin, isSupabaseConfigured } = await import('../lib/supabase');
      if (isSupabaseConfigured()) {
        const db = getSupabaseAdmin();
        const { data, error } = await db
          .from('core_lessons')
          .select('day_number, topic, marketing_headline, marketing_tagline')
          .gte('day_number', startDay)
          .lte('day_number', endDay)
          .order('day_number');

        if (!error && data) {
          return jsonResponse({ source: 'supabase-admin', lessons: data, startDay, endDay });
        }
      }
    } catch (e) {
      // Fall through to static range below
    }

    // Static range fallback: always returns N entries so the calendar never appears empty
    const lessons = [];
    for (let d = startDay; d <= endDay; d++) {
      const staticLesson = STATIC_LESSONS[d] || STATIC_LESSONS[((d - 1) % 7) + 1];
      lessons.push({
        day_number: d,
        topic: staticLesson?.topic || `Day ${d} Lesson`,
        marketing_headline: staticLesson?.topic || `Day ${d} Lesson`,
        marketing_tagline: '',
      });
    }
    return jsonResponse({ source: 'api-static-range', lessons, startDay, endDay });
  }

  if (Number.isNaN(day) || day < 1 || day > 365) {
    return jsonResponse(
      {
        error: 'Invalid day number',
        message: 'Day must be between 1 and 365',
      },
      400
    );
  }

  // Try to serve from static data (cycle through 1..7)
  const staticLesson = STATIC_LESSONS[day] || STATIC_LESSONS[((day - 1) % 7) + 1];

  if (staticLesson) {
    return jsonResponse({
      source: 'api-static',
      lesson: {
        id: `api-${day}`,
        day_number: day,
        topic: staticLesson.topic,
        universal_truth: staticLesson.universal_truth,
        marketing_headline: staticLesson.topic,
      },
      atoms: [
        { phase: 'Hook', content: { script: `Today we're learning about ${staticLesson.topic}!` } },
        { phase: 'Fact1', content: { script: staticLesson.universal_truth } },
        { phase: 'Fact2', content: { script: "Here's something interesting about this topic..." } },
        { phase: 'Fact3', content: { script: 'And one more fascinating fact...' } },
        { phase: 'Wisdom', content: { script: staticLesson.universal_truth } },
      ],
      shards: [],
      dayNumber: day,
    });
  }

  // Fallback for any day (should be unreachable, but keep it bulletproof)
  return jsonResponse({
    source: 'api-fallback',
    lesson: {
      id: `fallback-${day}`,
      day_number: day,
      topic: 'Daily Discovery',
      universal_truth: 'Every day brings something new to learn.',
      marketing_headline: 'Discover something amazing',
    },
    atoms: [
      { phase: 'Hook', content: { script: "Welcome to today's learning adventure!" } },
      { phase: 'Fact1', content: { script: "Let's explore together." } },
      { phase: 'Fact2', content: { script: "Here's something interesting..." } },
      { phase: 'Fact3', content: { script: 'And one more thing...' } },
      { phase: 'Wisdom', content: { script: 'Knowledge is power.' } },
    ],
    shards: [],
    dayNumber: day,
  });
}

export const config = {
  runtime: 'edge',
};


