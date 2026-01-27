/**
 * Video Jobs Submit API
 * 
 * POST /api/video-jobs/submit
 * Body: { job_id } OR { engine, status: 'queued', limit: 10 }
 * Submits job(s) to the engine API
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import { getEngine, ENGINE_TYPES } from '../../lib/engines';
import type { SubmitJobsRequest, VideoJob, EngineType } from '../../lib/engines/types';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const body = req.body as SubmitJobsRequest;
    const { job_id, engine, status = 'queued', limit = 10, dry_run = false } = body;
    
    const supabase = getSupabaseAdmin();
    
    // Build query
    let query = supabase
      .from('video_jobs')
      .select('*')
      .eq('status', status)
      .order('priority', { ascending: false })
      .order('created_at', { ascending: true })
      .limit(limit);
    
    if (job_id) {
      query = supabase.from('video_jobs').select('*').eq('id', job_id);
    } else if (engine) {
      query = query.eq('engine', engine);
    }
    
    const { data: jobs, error: fetchError } = await query;
    
    if (fetchError) {
      console.error('Fetch jobs error:', fetchError);
      return res.status(500).json({ error: 'Failed to fetch jobs', details: fetchError.message });
    }
    
    if (!jobs || jobs.length === 0) {
      return res.status(200).json({
        submitted: [],
        message: 'No jobs to submit',
        filters: { job_id, engine, status, limit },
      });
    }
    
    // Dry run - just report what would be submitted
    if (dry_run) {
      return res.status(200).json({
        dry_run: true,
        would_submit: jobs.length,
        jobs: jobs.map((j: VideoJob) => ({
          id: j.id,
          engine: j.engine,
          day: j.day_of_year,
          phase: j.phase,
        })),
      });
    }
    
    // Submit jobs
    const results: {
      submitted: string[];
      errors: Array<{ job_id: string; error: string }>;
    } = {
      submitted: [],
      errors: [],
    };
    
    for (const job of jobs as VideoJob[]) {
      try {
        const engineAdapter = getEngine(job.engine);
        
        // Check if engine is available
        const isAvailable = await engineAdapter.isAvailable();
        if (!isAvailable) {
          // Mark as blocked instead of failing
          await supabase
            .from('video_jobs')
            .update({
              status: 'blocked',
              error_message: `Engine ${job.engine} is not available`,
              updated_at: new Date().toISOString(),
            })
            .eq('id', job.id);
          
          results.errors.push({
            job_id: job.id,
            error: `Engine ${job.engine} not available - marked as blocked`,
          });
          continue;
        }
        
        // Submit to engine
        const submitResult = await engineAdapter.submit(job);
        
        // Update job status
        await supabase
          .from('video_jobs')
          .update({
            status: 'submitted',
            external_id: submitResult.external_id,
            submitted_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        results.submitted.push(job.id);
        
      } catch (error: any) {
        const errorMessage = error.message || 'Unknown error';
        
        // Check for blocked status (e.g., HeyGen 401)
        const isBlocked = errorMessage.includes('BLOCKED:401');
        
        await supabase
          .from('video_jobs')
          .update({
            status: isBlocked ? 'blocked' : 'failed',
            error_message: errorMessage,
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        results.errors.push({
          job_id: job.id,
          error: errorMessage,
        });
      }
    }
    
    return res.status(200).json({
      submitted: results.submitted,
      submitted_count: results.submitted.length,
      errors: results.errors,
      error_count: results.errors.length,
    });
    
  } catch (error) {
    console.error('Submit jobs error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
