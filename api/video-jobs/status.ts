/**
 * Video Jobs Status API
 * 
 * GET /api/video-jobs/status
 * Query params: day, phase, engine, status (all optional)
 * Returns: Array of jobs matching filters with summary
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import type { JobStatus, EngineType, Phase, VideoJob, JobStatusResponse } from '../../lib/engines/types';

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
  
  try {
    const supabase = getSupabaseAdmin();
    
    const { day, phase, engine, status, limit = '100' } = req.query;
    
    let query = supabase
      .from('video_jobs')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(parseInt(limit as string));
    
    // Apply filters
    if (day) {
      query = query.eq('day_of_year', parseInt(day as string));
    }
    if (phase) {
      query = query.eq('phase', phase as Phase);
    }
    if (engine) {
      query = query.eq('engine', engine as EngineType);
    }
    if (status) {
      query = query.eq('status', status as JobStatus);
    }
    
    const { data: jobs, error } = await query;
    
    if (error) {
      console.error('Video jobs status error:', error);
      return res.status(500).json({ error: 'Database error', details: error.message });
    }
    
    // Calculate summary
    const summary = {
      queued: 0,
      submitted: 0,
      processing: 0,
      completed: 0,
      failed: 0,
      blocked: 0,
    };
    
    (jobs || []).forEach((job: VideoJob) => {
      if (job.status in summary) {
        summary[job.status as keyof typeof summary]++;
      }
    });
    
    const response: JobStatusResponse = {
      jobs: jobs || [],
      total: jobs?.length || 0,
      summary,
    };
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('Video jobs status error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
