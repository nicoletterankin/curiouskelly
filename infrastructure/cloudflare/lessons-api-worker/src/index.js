/**
 * Curious Kelly Lessons API Worker
 * 
 * Serves lesson data from Cloudflare D1 as a Supabase mirror.
 * Provides edge-cached lesson access for redundancy and speed.
 * 
 * Routes:
 *   GET /health - Database health check
 *   GET /lesson/:day - Get lesson with atoms/shards
 *   GET /lessons - List all lessons
 *   GET /sync/status - Last sync status
 */

// Allowed origins for CORS
const ALLOWED_ORIGINS = [
  'https://curiouskelly.com',
  'https://www.curiouskelly.com',
  'https://learn.curiouskelly.com',
  'http://localhost:3000',
  'http://localhost:4321',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:4321',
];

/**
 * Get CORS headers based on request origin
 */
function getCorsHeaders(request) {
  const origin = request.headers.get('Origin');
  
  const isAllowed = !origin || ALLOWED_ORIGINS.some(allowed => {
    if (allowed.includes('localhost') || allowed.includes('127.0.0.1')) {
      return origin.startsWith('http://localhost:') || origin.startsWith('http://127.0.0.1:');
    }
    return origin === allowed;
  });

  return {
    'Access-Control-Allow-Origin': isAllowed ? (origin || '*') : ALLOWED_ORIGINS[0],
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json',
    'Cache-Control': 'public, max-age=300', // 5 minute cache
  };
}

/**
 * Parse JSON safely from D1 text fields
 */
function safeJsonParse(text, fallback = null) {
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const corsHeaders = getCorsHeaders(request);
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { 
        status: 204,
        headers: { ...corsHeaders, 'Access-Control-Max-Age': '86400' }
      });
    }

    // Only allow GET requests
    if (request.method !== 'GET') {
      return Response.json({ error: 'Method not allowed' }, { 
        status: 405, 
        headers: corsHeaders 
      });
    }

    try {
      // Route: Health check
      if (url.pathname === '/health') {
        return await handleHealth(env.DB, corsHeaders);
      }

      // Route: Get lesson by day
      const lessonMatch = url.pathname.match(/^\/lesson\/(\d+)$/);
      if (lessonMatch) {
        const dayNumber = parseInt(lessonMatch[1]);
        const archetype = url.searchParams.get('archetype') || 'The Scientist';
        const region = url.searchParams.get('region') || 'adult';
        return await handleGetLesson(env.DB, dayNumber, archetype, region, corsHeaders);
      }

      // Route: List all lessons
      if (url.pathname === '/lessons') {
        const limit = parseInt(url.searchParams.get('limit')) || 365;
        const offset = parseInt(url.searchParams.get('offset')) || 0;
        return await handleListLessons(env.DB, limit, offset, corsHeaders);
      }

      // Route: Sync status
      if (url.pathname === '/sync/status') {
        return await handleSyncStatus(env.DB, corsHeaders);
      }

      // Route: Root - API info
      if (url.pathname === '/' || url.pathname === '') {
        return Response.json({
          name: 'Curious Kelly Lessons API',
          version: '1.0.0',
          source: 'cloudflare-d1',
          routes: [
            'GET /health - Database health check',
            'GET /lesson/:day - Get lesson with atoms/shards',
            'GET /lessons - List all lessons',
            'GET /sync/status - Last sync status'
          ],
          documentation: 'https://docs.curiouskelly.com/api'
        }, { headers: corsHeaders });
      }

      // 404 for unknown routes
      return Response.json({ 
        error: 'Not found',
        routes: ['/health', '/lesson/:day', '/lessons', '/sync/status']
      }, { status: 404, headers: corsHeaders });

    } catch (error) {
      console.error('Worker error:', error);
      return Response.json({ 
        error: 'Internal server error',
        message: error.message,
        source: 'cloudflare-d1'
      }, { status: 500, headers: corsHeaders });
    }
  }
};

/**
 * Health check endpoint
 */
async function handleHealth(db, headers) {
  try {
    const result = await db.prepare('SELECT COUNT(*) as count FROM lessons').first();
    const syncMeta = await db.prepare('SELECT * FROM sync_metadata WHERE id = 1').first();
    
    return Response.json({
      status: 'healthy',
      source: 'cloudflare-d1',
      database: 'kelly-lessons-mirror',
      lessons: result?.count || 0,
      lastSync: syncMeta?.last_sync_at || null,
      timestamp: new Date().toISOString()
    }, { headers });
  } catch (error) {
    return Response.json({
      status: 'unhealthy',
      error: error.message,
      source: 'cloudflare-d1'
    }, { status: 500, headers });
  }
}

/**
 * Get lesson by day number with atoms and shards
 */
async function handleGetLesson(db, dayNumber, archetype, region, headers) {
  // Validate day number
  if (dayNumber < 1 || dayNumber > 365) {
    return Response.json({ 
      error: 'Invalid day number. Must be between 1 and 365.' 
    }, { status: 400, headers });
  }

  // Get lesson
  const lesson = await db.prepare(
    'SELECT * FROM lessons WHERE day_number = ?'
  ).bind(dayNumber).first();
  
  if (!lesson) {
    return Response.json({ 
      error: 'Lesson not found',
      dayNumber 
    }, { status: 404, headers });
  }

  // Parse JSON fields
  lesson.quick_quiz_questions = safeJsonParse(lesson.quick_quiz_questions, []);
  lesson.reflection_prompts = safeJsonParse(lesson.reflection_prompts, []);

  // Get atoms for this archetype
  const atomsResult = await db.prepare(
    `SELECT * FROM atoms 
     WHERE lesson_day = ? AND archetype = ? AND is_active = 1
     ORDER BY phase`
  ).bind(dayNumber, archetype).all();

  const atoms = (atomsResult.results || []).map(atom => ({
    ...atom,
    content: safeJsonParse(atom.content)
  }));

  // Get shards for this archetype and region
  const shardsResult = await db.prepare(
    `SELECT * FROM shards 
     WHERE lesson_day = ? AND (archetype = ? OR archetype IS NULL) AND region = ?`
  ).bind(dayNumber, archetype, region).all();

  const shards = (shardsResult.results || []).map(shard => ({
    ...shard,
    script_content: safeJsonParse(shard.script_content)
  }));

  // Build response with convenience fields
  return Response.json({
    lesson: {
      ...lesson,
      // Add virtual id field matching Supabase format
      id: lesson.id || `d1-${dayNumber}`
    },
    atoms,
    shards,
    meta: {
      dayNumber,
      archetype,
      region,
      atomsCount: atoms.length,
      shardsCount: shards.length
    },
    source: 'cloudflare-d1'
  }, { headers });
}

/**
 * List all lessons
 */
async function handleListLessons(db, limit, offset, headers) {
  const lessons = await db.prepare(
    `SELECT day_number, title, topic, subtitle, category, difficulty, duration_estimate
     FROM lessons 
     ORDER BY day_number
     LIMIT ? OFFSET ?`
  ).bind(limit, offset).all();

  const total = await db.prepare('SELECT COUNT(*) as count FROM lessons').first();

  return Response.json({
    lessons: lessons.results || [],
    count: (lessons.results || []).length,
    total: total?.count || 0,
    limit,
    offset,
    source: 'cloudflare-d1'
  }, { headers });
}

/**
 * Get sync status
 */
async function handleSyncStatus(db, headers) {
  const meta = await db.prepare('SELECT * FROM sync_metadata WHERE id = 1').first();
  
  if (!meta) {
    return Response.json({
      synced: false,
      message: 'No sync data available'
    }, { headers });
  }

  return Response.json({
    synced: true,
    lastSyncAt: meta.last_sync_at,
    counts: {
      lessons: meta.lessons_count,
      atoms: meta.atoms_count,
      shards: meta.shards_count
    },
    syncSource: meta.sync_source,
    syncDurationMs: meta.sync_duration_ms,
    source: 'cloudflare-d1'
  }, { headers });
}








