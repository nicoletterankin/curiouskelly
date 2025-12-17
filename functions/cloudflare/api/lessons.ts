/**
 * Cloudflare D1 Lessons Mirror API
 * 
 * This worker provides a D1 mirror of the lessons database
 * as a fallback when Supabase is unavailable.
 * 
 * Endpoint: GET /api/lessons/:dayNumber
 * Query params: archetype, ageBucket
 */

export interface Env {
  DB: D1Database;
}

interface LessonRow {
  id: number;
  day_number: number;
  topic: string;
  universal_truth: string;
  marketing_headline: string;
  marketing_tagline: string;
  marketing_pitch: string;
}

interface AtomRow {
  id: number;
  core_lesson_id: number;
  archetype: string;
  phase: string;
  content: string;
  hd_video_url: string | null;
  visual_url: string | null;
}

interface ShardRow {
  id: number;
  core_lesson_id: number;
  archetype: string;
  region: string;
  age: number | null;
  script_content: string;
}

export async function onRequest(context: { request: Request; env: Env; params: { dayNumber?: string } }): Promise<Response> {
  const { request, env, params } = context;
  
  // CORS headers
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Cache-Control': 'public, max-age=300', // 5 min cache
  };
  
  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }
  
  try {
    const url = new URL(request.url);
    const pathParts = url.pathname.split('/');
    const dayNumber = parseInt(pathParts[pathParts.length - 1]) || parseInt(params.dayNumber || '1');
    
    const archetype = url.searchParams.get('archetype') || 'The Scientist';
    const ageBucket = url.searchParams.get('ageBucket') || 'adult';
    
    // Check if D1 is available
    if (!env.DB) {
      return new Response(JSON.stringify({
        error: 'D1 not configured',
        message: 'Database mirror not available'
      }), {
        status: 503,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
    
    // Fetch lesson from D1
    const lesson = await env.DB.prepare(`
      SELECT id, day_number, topic, universal_truth, 
             marketing_headline, marketing_tagline, marketing_pitch
      FROM core_lessons
      WHERE day_number = ?
    `).bind(dayNumber).first<LessonRow>();
    
    if (!lesson) {
      return new Response(JSON.stringify({
        error: 'Lesson not found',
        dayNumber
      }), {
        status: 404,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
    
    // Fetch atoms for this lesson and archetype
    const atoms = await env.DB.prepare(`
      SELECT id, core_lesson_id, archetype, phase, content, hd_video_url, visual_url
      FROM lesson_atoms
      WHERE core_lesson_id = ? AND archetype = ?
      ORDER BY phase
    `).bind(lesson.id, archetype).all<AtomRow>();
    
    // Fetch shards for this lesson and archetype
    const shards = await env.DB.prepare(`
      SELECT id, core_lesson_id, archetype, region, age, script_content
      FROM lesson_shards
      WHERE core_lesson_id = ? AND archetype = ?
    `).bind(lesson.id, archetype).all<ShardRow>();
    
    // Parse JSON content fields
    const parsedAtoms = (atoms.results || []).map(atom => ({
      ...atom,
      content: typeof atom.content === 'string' ? JSON.parse(atom.content) : atom.content
    }));
    
    const parsedShards = (shards.results || []).map(shard => ({
      ...shard,
      script_content: typeof shard.script_content === 'string' ? JSON.parse(shard.script_content) : shard.script_content
    }));
    
    return new Response(JSON.stringify({
      source: 'd1',
      lesson,
      atoms: parsedAtoms,
      shards: parsedShards,
      dayNumber,
      archetype,
      ageBucket
    }), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
    
  } catch (error) {
    console.error('D1 Lessons API error:', error);
    
    return new Response(JSON.stringify({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

// Cloudflare Pages Functions export
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return onRequest({ request, env, params: {} });
  }
};








