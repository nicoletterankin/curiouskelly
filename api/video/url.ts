/**
 * Video URL API
 * 
 * GET /api/video/url
 * Query params: day, phase, age
 * Returns: Best approved video URL for production
 * Falls back to any completed video if none approved
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import type { EngineType, Phase, AgeCategory, VideoUrlResponse } from '../../lib/engines/types';

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
  
  const { day, phase, age = 'adult', source } = req.query;
  
  if (!day || !phase) {
    return res.status(400).json({
      error: 'Missing required parameters',
      required: ['day', 'phase'],
      optional: ['age (default: adult)', 'source (specific engine)'],
      example: '/api/video/url?day=20&phase=hook&age=adult',
    });
  }
  
  try {
    const supabase = getSupabaseAdmin();
    
    const dayNum = parseInt(day as string);
    const phaseStr = phase as Phase;
    const ageStr = (age as AgeCategory) || 'adult';
    const preferredEngine = source as EngineType | undefined;
    
    // Build query
    let query = supabase
      .from('video_jobs')
      .select('*')
      .eq('day_of_year', dayNum)
      .eq('phase', phaseStr)
      .eq('age_category', ageStr)
      .eq('status', 'completed')
      .not('output_url', 'is', null);
    
    // If specific engine requested
    if (preferredEngine) {
      query = query.eq('engine', preferredEngine);
    }
    
    // Order by: approved first, then quality score
    query = query
      .order('is_approved', { ascending: false })
      .order('quality_score', { ascending: false, nullsFirst: false })
      .order('completed_at', { ascending: false });
    
    const { data: jobs, error } = await query;
    
    if (error) {
      console.error('Video URL error:', error);
      return res.status(500).json({ error: 'Database error', details: error.message });
    }
    
    // Get available engines (all completed jobs)
    const availableEngines = [...new Set((jobs || []).map(j => j.engine))] as EngineType[];
    
    // No completed videos
    if (!jobs || jobs.length === 0) {
      // Check if there are any jobs at all
      const { data: pendingJobs } = await supabase
        .from('video_jobs')
        .select('engine, status')
        .eq('day_of_year', dayNum)
        .eq('phase', phaseStr)
        .eq('age_category', ageStr);
      
      const pendingEngines = [...new Set((pendingJobs || []).map(j => j.engine))];
      const pendingStatuses = [...new Set((pendingJobs || []).map(j => j.status))];
      
      const response: VideoUrlResponse = {
        url: null,
        engine: null,
        quality_score: null,
        is_approved: false,
        fallback: false,
        available_engines: [],
      };
      
      return res.status(200).json({
        ...response,
        status: 'no_videos',
        pending: {
          engines: pendingEngines,
          statuses: pendingStatuses,
        },
        hint: pendingJobs?.length 
          ? 'Videos are being generated. Check back soon.'
          : 'No video jobs exist for this lesson. Queue jobs first.',
      });
    }
    
    // Find best video
    // Priority: approved > highest quality score > most recent
    const approvedJob = jobs.find(j => j.is_approved);
    const bestJob = approvedJob || jobs[0];
    
    const response: VideoUrlResponse = {
      url: bestJob.output_url,
      engine: bestJob.engine,
      quality_score: bestJob.quality_score,
      is_approved: bestJob.is_approved || false,
      fallback: !approvedJob && jobs.length > 0,
      available_engines: availableEngines,
    };
    
    // Add metadata
    return res.status(200).json({
      ...response,
      day: dayNum,
      phase: phaseStr,
      age: ageStr,
      job_id: bestJob.id,
      alternatives: jobs.length > 1 ? jobs.slice(1).map(j => ({
        engine: j.engine,
        quality_score: j.quality_score,
        is_approved: j.is_approved,
        url: j.output_url,
      })) : [],
    });
    
  } catch (error) {
    console.error('Video URL error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
