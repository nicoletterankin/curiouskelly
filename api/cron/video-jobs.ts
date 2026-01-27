/**
 * Video Jobs Cron Handler
 * 
 * GET /api/cron/video-jobs
 * Runs the video job processing cycle
 * Should be called by Vercel Cron every minute
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { processQueuedJobs, checkPendingJobs, getQueueStats } from '../../lib/job-processor';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret if configured
  const cronSecret = process.env.CRON_SECRET;
  if (cronSecret && req.headers.authorization !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  
  // Also allow manual trigger with query param
  const { action, engine, limit } = req.query;
  
  try {
    const limitNum = limit ? parseInt(limit as string) : 10;
    const engineFilter = engine as string | undefined;
    
    const results: Record<string, any> = {
      timestamp: new Date().toISOString(),
    };
    
    // Run requested action(s)
    if (!action || action === 'all' || action === 'queued') {
      results.queued = await processQueuedJobs(engineFilter as any, limitNum);
    }
    
    if (!action || action === 'all' || action === 'pending') {
      results.pending = await checkPendingJobs(engineFilter as any, limitNum * 5);
    }
    
    if (!action || action === 'all' || action === 'stats') {
      results.stats = await getQueueStats();
    }
    
    // Summary
    const totalProcessed = (results.queued?.processed || 0) + (results.pending?.processed || 0);
    const totalCompleted = (results.queued?.completed || 0) + (results.pending?.completed || 0);
    const totalFailed = (results.queued?.failed || 0) + (results.pending?.failed || 0);
    
    return res.status(200).json({
      success: true,
      summary: {
        processed: totalProcessed,
        completed: totalCompleted,
        failed: totalFailed,
      },
      ...results,
    });
    
  } catch (error) {
    console.error('Video jobs cron error:', error);
    return res.status(500).json({ 
      error: 'Processing failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
