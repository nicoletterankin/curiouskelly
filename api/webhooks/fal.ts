/**
 * Fal.ai Webhook Handler
 * 
 * POST /api/webhooks/fal
 * Receives completion notifications from Fal.ai
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';

interface FalWebhookPayload {
  request_id: string;
  status: 'COMPLETED' | 'FAILED';
  output?: {
    video?: { url: string };
    video_url?: string;
  };
  error?: string;
}

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
    const payload = req.body as FalWebhookPayload;
    
    if (!payload.request_id) {
      return res.status(400).json({ error: 'Missing request_id in payload' });
    }
    
    const { request_id, status, output, error } = payload;
    
    const supabase = getSupabaseAdmin();
    
    // Find job by external_id (check both fal engines)
    const { data: job, error: fetchError } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('external_id', request_id)
      .in('engine', ['fal_latentsync', 'fal_sadtalker'])
      .single();
    
    if (fetchError || !job) {
      console.log('Fal webhook: Job not found for request_id:', request_id);
      return res.status(200).json({ 
        received: true, 
        warning: 'Job not found',
        request_id,
      });
    }
    
    // Extract video URL from various possible locations
    const videoUrl = output?.video?.url || output?.video_url;
    
    // Update based on status
    if (status === 'COMPLETED' && videoUrl) {
      await supabase
        .from('video_jobs')
        .update({
          status: 'completed',
          output_url: videoUrl,
          completed_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      console.log('Fal webhook: Job completed:', job.id);
      
    } else if (status === 'FAILED') {
      await supabase
        .from('video_jobs')
        .update({
          status: 'failed',
          error_message: error || 'Fal.ai reported failure',
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      console.log('Fal webhook: Job failed:', job.id, error);
    }
    
    return res.status(200).json({ 
      received: true,
      job_id: job.id,
      status,
    });
    
  } catch (error) {
    console.error('Fal webhook error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
