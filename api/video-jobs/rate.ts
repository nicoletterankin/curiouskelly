/**
 * Video Jobs Rate API
 * 
 * POST /api/video-jobs/rate
 * Body: { job_id, quality_score (1-10), quality_notes?, is_approved? }
 * Updates quality assessment
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import type { RateJobRequest } from '../../lib/engines/types';

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
    const body = req.body as RateJobRequest;
    const { job_id, quality_score, quality_notes, is_approved } = body;
    
    // Validate required fields
    if (!job_id) {
      return res.status(400).json({ error: 'job_id is required' });
    }
    
    if (quality_score === undefined || quality_score === null) {
      return res.status(400).json({ error: 'quality_score is required' });
    }
    
    if (quality_score < 1 || quality_score > 10) {
      return res.status(400).json({ error: 'quality_score must be between 1 and 10' });
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
    
    // Only rate completed jobs
    if (job.status !== 'completed') {
      return res.status(400).json({
        error: 'Can only rate completed jobs',
        current_status: job.status,
      });
    }
    
    // Update the job
    const updateData: Record<string, any> = {
      quality_score,
      updated_at: new Date().toISOString(),
    };
    
    if (quality_notes !== undefined) {
      updateData.quality_notes = quality_notes;
    }
    
    if (is_approved !== undefined) {
      updateData.is_approved = is_approved;
    }
    
    const { error: updateError } = await supabase
      .from('video_jobs')
      .update(updateData)
      .eq('id', job_id);
    
    if (updateError) {
      console.error('Rate job error:', updateError);
      return res.status(500).json({ error: 'Failed to update job', details: updateError.message });
    }
    
    return res.status(200).json({
      updated: true,
      job_id,
      quality_score,
      quality_notes,
      is_approved,
    });
    
  } catch (error) {
    console.error('Rate job error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
