/**
 * /api/motion-clip.ts
 * Fast lookup for completed motion clips during lesson playback.
 */

import { createClient } from '@supabase/supabase-js';

export const config = {
  runtime: 'edge',
};

function json(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      ...(init?.headers || {}),
    },
  });
}

export default async function handler(request: Request): Promise<Response> {
  if (request.method === 'OPTIONS') {
    return json(null, {
      status: 204,
      headers: { 'Cache-Control': 'public, max-age=0, s-maxage=0' },
    });
  }

  if (request.method !== 'GET') {
    return json({ error: 'Method not allowed' }, { status: 405 });
  }

  const url = new URL(request.url);
  const persona = url.searchParams.get('persona');
  const age = url.searchParams.get('age');
  const phase = url.searchParams.get('phase');

  if (!persona || !age || !phase) {
    return json(
      {
        error: 'Missing params',
        required: ['persona', 'age', 'phase'],
      },
      { status: 400 }
    );
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    return json(
      {
        error: 'Supabase is not configured on the server',
        required: ['PUBLIC_SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)', 'SUPABASE_SERVICE_ROLE_KEY'],
      },
      { status: 500 }
    );
  }

  const avatarKey = `${persona}_${age}`;

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey, {
      auth: { autoRefreshToken: false, persistSession: false },
      db: { schema: 'public' },
    });

    const { data, error } = await supabase
      .from('kelly_motion_library')
      .select('video_url, duration, status')
      .eq('avatar_key', avatarKey)
      .eq('phase', phase)
      .limit(1)
      .maybeSingle();

    if (error) {
      return json(
        { videoUrl: null, duration: null, fallback: true, error: error.message },
        {
          status: 200,
          headers: { 'Cache-Control': 'public, s-maxage=15, stale-while-revalidate=60' },
        }
      );
    }

    if (!data?.video_url || data.status !== 'completed') {
      return json(
        { videoUrl: null, duration: data?.duration ?? null, fallback: true },
        {
          status: 200,
          headers: { 'Cache-Control': 'public, s-maxage=30, stale-while-revalidate=300' },
        }
      );
    }

    return json(
      { videoUrl: data.video_url, duration: data.duration ?? null, fallback: false },
      {
        status: 200,
        headers: { 'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400' },
      }
    );
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    return json(
      { videoUrl: null, duration: null, fallback: true, error: message },
      {
        status: 500,
        headers: { 'Cache-Control': 'no-store' },
      }
    );
  }
}
