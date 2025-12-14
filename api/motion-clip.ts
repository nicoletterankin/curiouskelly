/**
 * /api/motion-clip.ts
 * Fast lookup for completed motion clips during lesson playback.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

function setCors(res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  setCors(res);

  if (req.method === 'OPTIONS') {
    res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=0');
    return res.status(204).send('');
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const persona = (req.query?.persona as string | undefined) || null;
  const age = (req.query?.age as string | undefined) || null;
  const phase = (req.query?.phase as string | undefined) || null;

  if (!persona || !age || !phase) {
    return res.status(400).json({
      error: 'Missing params',
      required: ['persona', 'age', 'phase'],
    });
  }

  if (!isSupabaseConfigured()) {
    return res.status(500).json({
      error: 'Supabase is not configured on the server',
      required: ['PUBLIC_SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)', 'SUPABASE_SERVICE_ROLE_KEY'],
    });
  }

  const avatarKey = `${persona}_${age}`;

  try {
    const supabase = getSupabaseAdmin();

    const { data, error } = await supabase
      .from('kelly_motion_library')
      .select('video_url, duration, status')
      .eq('avatar_key', avatarKey)
      .eq('phase', phase)
      .limit(1)
      .maybeSingle();

    if (error) {
      res.setHeader('Cache-Control', 'public, s-maxage=15, stale-while-revalidate=60');
      return res.status(200).json({ videoUrl: null, duration: null, fallback: true, error: error.message });
    }

    if (!data?.video_url || data.status !== 'completed') {
      res.setHeader('Cache-Control', 'public, s-maxage=30, stale-while-revalidate=300');
      return res.status(200).json({ videoUrl: null, duration: data?.duration ?? null, fallback: true });
    }

    res.setHeader('Cache-Control', 'public, s-maxage=3600, stale-while-revalidate=86400');
    return res.status(200).json({ videoUrl: data.video_url, duration: data.duration ?? null, fallback: false });
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    res.setHeader('Cache-Control', 'no-store');
    return res.status(500).json({ videoUrl: null, duration: null, fallback: true, error: message });
  }
}
