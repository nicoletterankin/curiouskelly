/**
 * Retry HeyGen Jobs API
 * 
 * POST /api/video-jobs/retry-heygen
 * No body needed
 * Unblocks and resubmits all blocked HeyGen jobs
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import { getEngine } from '../../lib/engines';

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
    const supabase = getSupabaseAdmin();
    
    // First check if HeyGen is available
    const heygenAdapter = getEngine('heygen');
    const isAvailable = await heygenAdapter.isAvailable();
    
    if (!isAvailable) {
      return res.status(503).json({
        error: 'HeyGen API is still not available',
        hint: 'Check HEYGEN_API_KEY or try again later',
        status: 'blocked',
      });
    }
    
    // Call the unblock_heygen_jobs function if it exists
    // This is a database function that Antigravity may have created
    try {
      const { data: unblockResult, error: rpcError } = await supabase
        .rpc('unblock_heygen_jobs');
      
      if (rpcError) {
        console.log('unblock_heygen_jobs RPC not available:', rpcError.message);
      }
    } catch {
      // Function may not exist yet
    }
    
    // Get all blocked HeyGen jobs
    const { data: blockedJobs, error: fetchError } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('engine', 'heygen')
      .eq('status', 'blocked')
      .order('priority', { ascending: false })
      .order('created_at', { ascending: true });
    
    if (fetchError) {
      console.error('Fetch blocked jobs error:', fetchError);
      return res.status(500).json({ error: 'Failed to fetch blocked jobs' });
    }
    
    if (!blockedJobs || blockedJobs.length === 0) {
      return res.status(200).json({
        message: 'No blocked HeyGen jobs to retry',
        unblocked: 0,
        submitted: 0,
      });
    }
    
    // Unblock and submit jobs
    const results = {
      unblocked: 0,
      submitted: 0,
      errors: [] as Array<{ job_id: string; error: string }>,
    };
    
    for (const job of blockedJobs) {
      try {
        // Update status to queued first
        await supabase
          .from('video_jobs')
          .update({
            status: 'queued',
            error_message: null,
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        results.unblocked++;
        
        // Try to submit
        const submitResult = await heygenAdapter.submit(job);
        
        await supabase
          .from('video_jobs')
          .update({
            status: 'submitted',
            external_id: submitResult.external_id,
            submitted_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        results.submitted++;
        
      } catch (error: any) {
        const errorMessage = error.message || 'Unknown error';
        const isStillBlocked = errorMessage.includes('BLOCKED:401');
        
        await supabase
          .from('video_jobs')
          .update({
            status: isStillBlocked ? 'blocked' : 'failed',
            error_message: errorMessage,
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        results.errors.push({ job_id: job.id, error: errorMessage });
        
        // If still blocked, stop trying
        if (isStillBlocked) {
          return res.status(503).json({
            error: 'HeyGen API returned 401 - still blocked',
            unblocked: results.unblocked,
            submitted: results.submitted,
            errors: results.errors,
          });
        }
      }
    }
    
    return res.status(200).json({
      success: true,
      unblocked: results.unblocked,
      submitted: results.submitted,
      errors: results.errors,
    });
    
  } catch (error) {
    console.error('Retry HeyGen error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
