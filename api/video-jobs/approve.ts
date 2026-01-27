/**
 * Video Jobs Approve API
 * 
 * POST /api/video-jobs/approve
 * Body: { job_id }
 * Marks video as approved for production
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';

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
    const { job_id } = req.body;
    
    if (!job_id) {
      return res.status(400).json({ error: 'job_id is required' });
    }
    
    const supabase = getSupabaseAdmin();
    
    // Get the job
    const { data: job, error: fetchError } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('id', job_id)
      .single();
    
    if (fetchError || !job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    // Only approve completed jobs with output
    if (job.status !== 'completed') {
      return res.status(400).json({
        error: 'Can only approve completed jobs',
        current_status: job.status,
      });
    }
    
    if (!job.output_url) {
      return res.status(400).json({
        error: 'Job has no output_url',
        hint: 'Wait for video generation to complete',
      });
    }
    
    // Update the job
    const { error: updateError } = await supabase
      .from('video_jobs')
      .update({
        is_approved: true,
        updated_at: new Date().toISOString(),
      })
      .eq('id', job_id);
    
    if (updateError) {
      console.error('Approve job error:', updateError);
      return res.status(500).json({ error: 'Failed to approve job', details: updateError.message });
    }
    
    // If there's a production_videos view/table, we might want to insert there too
    // For now, the is_approved flag is sufficient
    
    return res.status(200).json({
      approved: true,
      job_id,
      output_url: job.output_url,
      day: job.day_of_year,
      phase: job.phase,
      engine: job.engine,
    });
    
  } catch (error) {
    console.error('Approve job error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
