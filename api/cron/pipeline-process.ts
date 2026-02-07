/**
 * Pipeline Processing Cron
 * 
 * Runs every minute to:
 * 1. Process queued jobs with provider fallback
 * 2. Check pending jobs for completion
 * 3. Run eval gates on completed videos
 * 
 * Vercel cron: 0/1 * * * * (every minute)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { runProcessingCycle, getQueueStats } from '../../lib/fallback-queue';
import { notifyDailySummary } from '../../lib/email-alerts';

// Cron secret for security
const CRON_SECRET = process.env.CRON_SECRET;

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // Verify cron secret if configured
  if (CRON_SECRET && req.headers.authorization !== `Bearer ${CRON_SECRET}`) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  
  const startTime = Date.now();
  
  try {
    // Run processing cycle
    const results = await runProcessingCycle({
      enableEvalGates: true,
      enableAlerts: true,
    });
    
    const elapsed = Date.now() - startTime;
    
    // Check if we should send daily summary (at midnight UTC)
    const now = new Date();
    if (now.getUTCHours() === 0 && now.getUTCMinutes() === 0) {
      const dayOfYear = getDayOfYear(now);
      const stats = await getQueueStats();
      
      await notifyDailySummary(dayOfYear, {
        completed: stats.by_status.completed || 0,
        failed: stats.by_status.failed || 0,
        pending: (stats.by_status.queued || 0) + (stats.by_status.submitted || 0) + (stats.by_status.processing || 0),
        blocked: stats.by_status.blocked || 0,
      });
    }
    
    res.status(200).json({
      success: true,
      elapsed_ms: elapsed,
      queue_batch: {
        processed: results.queue_batch.processed,
        succeeded: results.queue_batch.succeeded,
        failed: results.queue_batch.failed,
      },
      pending_check: {
        checked: results.pending_check.checked,
        completed: results.pending_check.completed,
        still_processing: results.pending_check.still_processing,
      },
      provider_status: results.stats.provider_availability,
    });
    
  } catch (error: any) {
    console.error('Cron error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      elapsed_ms: Date.now() - startTime,
    });
  }
}

function getDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

// Export for Vercel cron config
export const config = {
  schedule: '* * * * *', // Every minute
};
