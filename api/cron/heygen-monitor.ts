/**
 * HEYGEN MONITOR CRON JOB
 * 
 * Checks HeyGen video status every 4 hours and logs performance metrics.
 * Runs via Vercel Cron.
 * 
 * Schedule: Every 4 hours (0 */4 * * *)
 */

import { createClient } from '@supabase/supabase-js';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface VideoStatus {
  video_id: string;
  status: string;
  video_url?: string;
  duration?: number;
  error?: string;
}

interface QueueData {
  videos: { [day: string]: string };
}

async function checkVideoStatus(videoId: string): Promise<VideoStatus> {
  try {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`,
      { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
    );
    
    const result = await response.json();
    
    return {
      video_id: videoId,
      status: result.data?.status || 'unknown',
      video_url: result.data?.video_url,
      duration: result.data?.duration,
      error: result.data?.error
    };
  } catch (err) {
    return { video_id: videoId, status: 'error', error: String(err) };
  }
}

export default async function handler(req: any, res: any) {
  // Verify this is a cron request
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    // Allow if no CRON_SECRET is set (for testing)
    if (process.env.CRON_SECRET) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }

  console.log('🎬 HeyGen Monitor starting...');

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  const now = new Date().toISOString();

  // Sample some videos to check (not all 365 to stay within rate limits)
  // Check a spread: day 1, 50, 100, 150, 200, 250, 300, 350, 365
  const sampleDays = [1, 50, 100, 150, 200, 250, 300, 350, 365];
  
  // Fetch the queue files via API or directly
  // For now, we'll use a simplified approach checking known IDs
  
  const results = {
    checked_at: now,
    sample_size: sampleDays.length,
    completed: 0,
    pending: 0,
    failed: 0,
    samples: [] as any[]
  };

  // Fetch queue data
  const months = [
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december'
  ];

  const videoMap: { [day: number]: string } = {};

  // Try to load queue data (this would need to be adapted based on how queues are stored)
  try {
    const baseUrl = process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}` 
      : 'https://curiouskelly.com';
    
    for (const month of months) {
      try {
        const resp = await fetch(`${baseUrl}/data/email-summary-video/${month}-video-queue.json`);
        if (resp.ok) {
          const data: QueueData = await resp.json();
          for (const [day, videoId] of Object.entries(data.videos || {})) {
            videoMap[parseInt(day)] = videoId;
          }
        }
      } catch {
        // Queue file doesn't exist or can't be fetched
      }
    }
  } catch (err) {
    console.log('Could not fetch queue files:', err);
  }

  // Check sample videos
  for (const day of sampleDays) {
    const videoId = videoMap[day];
    if (!videoId) continue;

    const status = await checkVideoStatus(videoId);
    
    if (status.status === 'completed') results.completed++;
    else if (status.status === 'failed') results.failed++;
    else results.pending++;

    results.samples.push({
      day,
      video_id: videoId,
      status: status.status,
      video_url: status.video_url,
      duration: status.duration
    });

    // Small delay to avoid rate limiting
    await new Promise(r => setTimeout(r, 100));
  }

  // Log to Supabase
  try {
    await supabase.from('heygen_performance_logs').insert({
      checked_at: now,
      completed_count: results.completed,
      pending_count: results.pending,
      failed_count: results.failed,
      sample_data: results.samples
    });
  } catch (err) {
    console.log('Could not log to Supabase:', err);
  }

  console.log(`✅ HeyGen Monitor complete: ${results.completed} completed, ${results.pending} pending, ${results.failed} failed`);

  return res.status(200).json({
    success: true,
    message: `Checked ${results.sample_size} sample videos`,
    results
  });
}
