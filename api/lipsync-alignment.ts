import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

/**
 * Kelly Lipsync Alignment API
 * 
 * Fetches pre-computed phoneme alignments and blendshape timelines
 * for a specific lesson segment.
 * 
 * GET /api/lipsync-alignment?day=1&age=6-12&lang=en&phase=script
 * 
 * Returns:
 * {
 *   words: [...],
 *   phones: [...],
 *   blendshapeTimeline: [...],
 *   duration: number,
 *   method: string,
 *   confidence: number
 * }
 */

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'public, max-age=31536000'); // Cache for 1 year

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Parse query parameters
    const {
      day,
      age,
      lang = 'en',
      phase = 'script',
    } = req.query;

    // Validate required params
    if (!day) {
      return res.status(400).json({ error: 'day parameter required' });
    }
    if (!age) {
      return res.status(400).json({ error: 'age parameter required' });
    }

    const dayNumber = parseInt(day as string, 10);
    const ageBucket = age as string;
    const language = lang as string;
    const phaseType = phase as string;

    if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
      return res.status(400).json({ error: 'day must be between 1 and 365' });
    }

    // Initialize Supabase
    if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
      return res.status(500).json({ error: 'Database not configured' });
    }

    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

    // Fetch alignment
    const { data, error } = await supabase
      .from('lipsync_alignments')
      .select('*')
      .eq('day_number', dayNumber)
      .eq('age_bucket', ageBucket)
      .eq('language', language)
      .eq('phase', phaseType)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        // No rows returned - try fallback to estimation
        return res.status(404).json({
          error: 'Alignment not found',
          suggestion: 'Use /api/align with transcript for on-demand generation',
          params: { day: dayNumber, age: ageBucket, lang: language, phase: phaseType },
        });
      }
      throw error;
    }

    // Return alignment data
    return res.status(200).json({
      words: data.words,
      phones: data.phones,
      blendshapeTimeline: data.blendshape_timeline,
      duration: data.duration_seconds,
      method: data.method,
      confidence: data.confidence,
      fps: data.fps,
      transcript: data.transcript,
      
      // Metadata
      meta: {
        day: data.day_number,
        ageBucket: data.age_bucket,
        language: data.language,
        phase: data.phase,
        createdAt: data.created_at,
      },
    });

  } catch (error) {
    console.error('[Lipsync API] Error:', error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : 'Internal server error',
    });
  }
}

