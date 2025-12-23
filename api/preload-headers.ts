/**
 * Preload Headers API
 * 
 * Returns preload headers for lesson routes to enable instant asset loading.
 * This is called by Edge Middleware or can be used directly in HTML.
 * 
 * Usage:
 *   GET /api/preload-headers?day=161&archetype=The Scientist
 */

export const config = {
  runtime: 'edge',
};

export default async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const searchParams = url.searchParams;
  const day = searchParams.get('day') || getTodayDayNumber();
  const archetype = searchParams.get('archetype') || 'The Scientist';
  
  const dayNum = parseInt(day);
  if (dayNum < 1 || dayNum > 365) {
    return new Response(
      JSON.stringify({ error: 'Invalid day number' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }
  
  const paddedDay = String(dayNum).padStart(3, '0');
  
  // Generate preload links
  const preloadLinks = [
    // Preload lesson metadata (Edge Config)
    `</api/lessons/${dayNum}?archetype=${encodeURIComponent(archetype)}>; rel=preload; as=fetch; crossorigin`,
    
    // Preload first phase video (Hook) - CRITICAL for zero buffering
    `</blob/videos/day-${paddedDay}/${archetype.toLowerCase().replace(/\s+/g, '-')}/hook.mp4>; rel=preload; as=video`,
    
    // Preload first phase audio
    `</blob/audio/day-${paddedDay}/${archetype.toLowerCase().replace(/\s+/g, '-')}/hook.mp3>; rel=preload; as=audio`,
    
    // Preload first phase visual
    `</blob/visuals/day-${paddedDay}/hook-infographic.png>; rel=preload; as=image`,
    
    // Preload next 2 phases (Question, Context) - ZERO BUFFERING
    `</blob/videos/day-${paddedDay}/${archetype.toLowerCase().replace(/\s+/g, '-')}/question.mp4>; rel=prefetch; as=video`,
    `</blob/videos/day-${paddedDay}/${archetype.toLowerCase().replace(/\s+/g, '-')}/context.mp4>; rel=prefetch; as=video`,
    
    // Preload adjacent days (for calendar navigation)
    `</api/lessons/${dayNum + 1}>; rel=prefetch; as=fetch`,
    `</api/lessons/${dayNum - 1}>; rel=prefetch; as=fetch`,
  ];
  
  return new Response(
    JSON.stringify({
      links: preloadLinks,
      day: dayNum,
      archetype,
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Link': preloadLinks.join(', '),
      },
    }
  );
}

function getTodayDayNumber(): string {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 1);
  const diff = now.getTime() - start.getTime();
  const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24)) + 1;
  return String(Math.min(365, Math.max(1, dayOfYear)));
}

