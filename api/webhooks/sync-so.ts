/**
 * Sync.so Webhook Handler
 * 
 * POST /api/webhooks/sync-so
 * Receives completion notifications from Sync Labs
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';

interface SyncSoWebhookPayload {
  id: string;
  status: 'COMPLETED' | 'FAILED' | 'REJECTED';
  output?: Array<{ url: string }>;
  outputUrl?: string;
  error?: string;
  message?: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Sync-Signature');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    // TODO: Verify webhook signature if Sync.so provides one
    // const signature = req.headers['x-sync-signature'];
    
    const payload = req.body as SyncSoWebhookPayload;
    
    if (!payload.id) {
      return res.status(400).json({ error: 'Missing id in payload' });
    }
    
    const { id, status, output, outputUrl, error, message } = payload;
    
    const supabase = getSupabaseAdmin();
    
    // Find job by external_id
    const { data: job, error: fetchError } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('external_id', id)
      .eq('engine', 'sync_so')
      .single();
    
    if (fetchError || !job) {
      console.log('Sync.so webhook: Job not found for id:', id);
      return res.status(200).json({ 
        received: true, 
        warning: 'Job not found',
        id,
      });
    }
    
    // Extract video URL
    const videoUrl = output?.[0]?.url || outputUrl;
    
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
      
      console.log('Sync.so webhook: Job completed:', job.id);
      
    } else if (status === 'FAILED' || status === 'REJECTED') {
      await supabase
        .from('video_jobs')
        .update({
          status: 'failed',
          error_message: error || message || 'Sync.so reported failure',
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      console.log('Sync.so webhook: Job failed:', job.id, error || message);
    }
    
    return res.status(200).json({ 
      received: true,
      job_id: job.id,
      status,
    });
    
  } catch (error) {
    console.error('Sync.so webhook error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
