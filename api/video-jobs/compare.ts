/**
 * Video Jobs Compare API
 * 
 * GET /api/video-jobs/compare
 * Query params: day, phase, age (all required)
 * Returns: All engines for that specific lesson, side-by-side
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import type { EngineType, Phase, AgeCategory, VideoJob, JobCompareResponse } from '../../lib/engines/types';
import { ENGINE_TYPES } from '../../lib/engines';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const { day, phase, age } = req.query;
  
  if (!day || !phase || !age) {
    return res.status(400).json({
      error: 'Missing required parameters',
      required: ['day', 'phase', 'age'],
      example: '/api/video-jobs/compare?day=20&phase=hook&age=adult',
    });
  }
  
  try {
    const supabase = getSupabaseAdmin();
    
    const dayNum = parseInt(day as string);
    const phaseStr = phase as Phase;
    const ageStr = age as AgeCategory;
    
    // Get all jobs for this day/phase/age across all engines
    const { data: jobs, error } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('day_of_year', dayNum)
      .eq('phase', phaseStr)
      .eq('age_category', ageStr)
      .order('created_at', { ascending: false });
    
    if (error) {
      console.error('Video jobs compare error:', error);
      return res.status(500).json({ error: 'Database error', details: error.message });
    }
    
    // Group by engine (take most recent job per engine)
    const engines: Record<EngineType, VideoJob | null> = {
      heygen: null,
      fal_latentsync: null,
      fal_sadtalker: null,
      sync_so: null,
      musetalk_local: null,
    };
    
    (jobs || []).forEach((job: VideoJob) => {
      if (!engines[job.engine]) {
        engines[job.engine] = job;
      }
    });
    
    const response: JobCompareResponse = {
      day: dayNum,
      phase: phaseStr,
      age: ageStr,
      engines,
    };
    
    // Add summary stats
    const stats = {
      total_engines: ENGINE_TYPES.length,
      engines_with_jobs: Object.values(engines).filter(e => e !== null).length,
      completed: Object.values(engines).filter(e => e?.status === 'completed').length,
      approved: Object.values(engines).filter(e => e?.is_approved).length,
      best_quality: (() => {
        const completed = Object.entries(engines)
          .filter(([_, job]) => job?.status === 'completed' && job.quality_score)
          .sort((a, b) => (b[1]?.quality_score || 0) - (a[1]?.quality_score || 0));
        return completed[0]?.[0] || null;
      })(),
    };
    
    return res.status(200).json({ ...response, stats });
    
  } catch (error) {
    console.error('Video jobs compare error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
