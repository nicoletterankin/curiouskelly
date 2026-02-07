/**
 * Pipeline Status API
 * GET /api/pipeline/status
 * 
 * Returns real-time pipeline status for the monitoring dashboard.
 * Includes provider availability, queue stats, and recent alerts.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from '../lib/supabase';
import { getEngineStatus, ENGINE_TYPES } from '../../lib/engines';
import { cors } from '../../lib/cors';

export interface ProviderStatus {
  name: string;
  displayName: string;
  status: 'available' | 'degraded' | 'unavailable';
  lastCheck: string;
  successRate: number;
  avgProcessingTime?: number;
}

export interface QueueStats {
  queued: number;
  submitted: number;
  processing: number;
  completed_today: number;
  failed_today: number;
  blocked: number;
}

export interface PipelineAlert {
  id: string;
  type: 'eval_failure' | 'job_failure' | 'pipeline_error' | 'provider_down';
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  job_id?: string;
  day_of_year?: number;
  phase?: string;
  timestamp: string;
}

export interface PipelineStatusResponse {
  timestamp: string;
  providers: ProviderStatus[];
  queue: QueueStats;
  alerts: PipelineAlert[];
  health: 'healthy' | 'degraded' | 'critical';
  daily_progress: {
    target_day: number;
    phases_complete: number;
    phases_total: number;
    percent: number;
  };
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS (public endpoint for dashboard)
  if (!cors(req, res, { allowAllOrigins: true })) return;
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).toISOString();
    
    // Get provider status
    const engineStatus = await getEngineStatus();
    const providers: ProviderStatus[] = [];
    
    for (const engineType of ENGINE_TYPES) {
      const status = engineStatus[engineType];
      providers.push({
        name: engineType,
        displayName: status?.displayName || engineType,
        status: status?.available ? 'available' : 'unavailable',
        lastCheck: now.toISOString(),
        successRate: 0, // Would need historical data
      });
    }
    
    // Get queue statistics
    let queue: QueueStats = {
      queued: 0,
      submitted: 0,
      processing: 0,
      completed_today: 0,
      failed_today: 0,
      blocked: 0,
    };
    
    let alerts: PipelineAlert[] = [];
    let daily_progress = {
      target_day: 1,
      phases_complete: 0,
      phases_total: 5,
      percent: 0,
    };
    
    if (isSupabaseConfigured()) {
      const supabase = getSupabaseAdmin();
      
      // Count by status
      const { data: jobs } = await supabase
        .from('video_jobs')
        .select('status, created_at, completed_at');
      
      if (jobs) {
        for (const job of jobs) {
          if (job.status === 'queued') queue.queued++;
          else if (job.status === 'submitted') queue.submitted++;
          else if (job.status === 'processing') queue.processing++;
          else if (job.status === 'blocked') queue.blocked++;
          else if (job.status === 'completed' && job.completed_at >= todayStart) queue.completed_today++;
          else if (job.status === 'failed' && job.created_at >= todayStart) queue.failed_today++;
        }
      }
      
      // Get recent alerts (from video_jobs with errors)
      const { data: failedJobs } = await supabase
        .from('video_jobs')
        .select('id, day_of_year, phase, engine, error_message, updated_at')
        .in('status', ['failed', 'eval_failed', 'blocked'])
        .gte('updated_at', new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString())
        .order('updated_at', { ascending: false })
        .limit(10);
      
      if (failedJobs) {
        alerts = failedJobs.map(job => ({
          id: job.id,
          type: job.error_message?.includes('eval') ? 'eval_failure' as const : 'job_failure' as const,
          severity: 'error' as const,
          message: job.error_message || 'Unknown error',
          job_id: job.id,
          day_of_year: job.day_of_year,
          phase: job.phase,
          timestamp: job.updated_at,
        }));
      }
      
      // Calculate daily progress (Day 1 for now)
      const { data: day1Jobs } = await supabase
        .from('video_jobs')
        .select('phase, status')
        .eq('day_of_year', 1)
        .eq('status', 'completed');
      
      daily_progress = {
        target_day: 1,
        phases_complete: day1Jobs?.length || 0,
        phases_total: 5, // hook, story, wonder, action, wisdom
        percent: Math.round(((day1Jobs?.length || 0) / 5) * 100),
      };
    }
    
    // Determine overall health
    const availableProviders = providers.filter(p => p.status === 'available').length;
    const criticalAlerts = alerts.filter(a => a.severity === 'critical').length;
    
    let health: PipelineStatusResponse['health'] = 'healthy';
    if (availableProviders === 0 || criticalAlerts > 0) {
      health = 'critical';
    } else if (availableProviders < 2 || alerts.length > 5) {
      health = 'degraded';
    }
    
    const response: PipelineStatusResponse = {
      timestamp: now.toISOString(),
      providers,
      queue,
      alerts,
      health,
      daily_progress,
    };
    
    // Cache for 30 seconds
    res.setHeader('Cache-Control', 's-maxage=30, stale-while-revalidate=60');
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('Pipeline status error:', error);
    return res.status(500).json({
      error: 'Failed to get pipeline status',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
