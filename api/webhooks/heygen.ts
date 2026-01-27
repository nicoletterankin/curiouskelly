/**
 * HeyGen Webhook Handler
 * 
 * POST /api/webhooks/heygen
 * Receives completion notifications from HeyGen
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';

interface HeyGenWebhookPayload {
  event_type: 'avatar_video.success' | 'avatar_video.fail';
  event_data: {
    video_id: string;
    status: string;
    video_url?: string;
    error?: string;
  };
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-HeyGen-Signature');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    // TODO: Verify webhook signature if HeyGen provides one
    // const signature = req.headers['x-heygen-signature'];
    
    const payload = req.body as HeyGenWebhookPayload;
    
    if (!payload.event_data?.video_id) {
      return res.status(400).json({ error: 'Missing video_id in payload' });
    }
    
    const { video_id, status, video_url, error } = payload.event_data;
    
    const supabase = getSupabaseAdmin();
    
    // Find job by external_id
    const { data: job, error: fetchError } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('external_id', video_id)
      .eq('engine', 'heygen')
      .single();
    
    if (fetchError || !job) {
      console.log('HeyGen webhook: Job not found for video_id:', video_id);
      return res.status(200).json({ 
        received: true, 
        warning: 'Job not found',
        video_id,
      });
    }
    
    // Update based on event type
    if (payload.event_type === 'avatar_video.success' && video_url) {
      await supabase
        .from('video_jobs')
        .update({
          status: 'completed',
          output_url: video_url,
          completed_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      console.log('HeyGen webhook: Job completed:', job.id);
      
    } else if (payload.event_type === 'avatar_video.fail') {
      await supabase
        .from('video_jobs')
        .update({
          status: 'failed',
          error_message: error || 'HeyGen reported failure',
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      console.log('HeyGen webhook: Job failed:', job.id, error);
    }
    
    return res.status(200).json({ 
      received: true,
      job_id: job.id,
      event_type: payload.event_type,
    });
    
  } catch (error) {
    console.error('HeyGen webhook error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
