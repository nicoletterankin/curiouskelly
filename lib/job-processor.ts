/**
 * Video Job Processor
 * 
 * Background processing for video generation jobs.
 * Can be run as a cron job or triggered manually.
 */

import { getSupabaseAdmin } from '../api/lib/supabase';
import { getEngine, ENGINE_TYPES } from './engines';
import type { VideoJob, EngineType, JobStatus } from './engines/types';

export interface ProcessResult {
  processed: number;
  submitted: number;
  completed: number;
  failed: number;
  errors: Array<{ job_id: string; error: string }>;
}

/**
 * Process queued jobs by submitting them to engines
 */
export async function processQueuedJobs(
  engine?: EngineType,
  limit = 10
): Promise<ProcessResult> {
  const supabase = getSupabaseAdmin();
  
  const result: ProcessResult = {
    processed: 0,
    submitted: 0,
    completed: 0,
    failed: 0,
    errors: [],
  };
  
  // Build query for queued jobs
  let query = supabase
    .from('video_jobs')
    .select('*')
    .eq('status', 'queued')
    .order('priority', { ascending: false })
    .order('created_at', { ascending: true })
    .limit(limit);
  
  if (engine) {
    query = query.eq('engine', engine);
  }
  
  const { data: jobs, error } = await query;
  
  if (error) {
    console.error('Process queued jobs error:', error);
    throw error;
  }
  
  if (!jobs || jobs.length === 0) {
    return result;
  }
  
  for (const job of jobs as VideoJob[]) {
    result.processed++;
    
    try {
      const engineAdapter = getEngine(job.engine);
      
      // Check availability
      const isAvailable = await engineAdapter.isAvailable();
      if (!isAvailable) {
        await supabase
          .from('video_jobs')
          .update({
            status: 'blocked',
            error_message: `Engine ${job.engine} not available`,
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        result.failed++;
        result.errors.push({ job_id: job.id, error: 'Engine not available' });
        continue;
      }
      
      // Submit job
      const submitResult = await engineAdapter.submit(job);
      
      await supabase
        .from('video_jobs')
        .update({
          status: 'submitted',
          external_id: submitResult.external_id,
          submitted_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      result.submitted++;
      
    } catch (error: any) {
      const errorMessage = error.message || 'Unknown error';
      const isBlocked = errorMessage.includes('BLOCKED:');
      
      await supabase
        .from('video_jobs')
        .update({
          status: isBlocked ? 'blocked' : 'failed',
          error_message: errorMessage,
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      result.failed++;
      result.errors.push({ job_id: job.id, error: errorMessage });
    }
  }
  
  return result;
}

/**
 * Check status of pending jobs (submitted/processing)
 */
export async function checkPendingJobs(
  engine?: EngineType,
  limit = 50
): Promise<ProcessResult> {
  const supabase = getSupabaseAdmin();
  
  const result: ProcessResult = {
    processed: 0,
    submitted: 0,
    completed: 0,
    failed: 0,
    errors: [],
  };
  
  // Get jobs that are submitted or processing
  let query = supabase
    .from('video_jobs')
    .select('*')
    .in('status', ['submitted', 'processing'])
    .not('external_id', 'is', null)
    .order('submitted_at', { ascending: true })
    .limit(limit);
  
  if (engine) {
    query = query.eq('engine', engine);
  }
  
  const { data: jobs, error } = await query;
  
  if (error) {
    console.error('Check pending jobs error:', error);
    throw error;
  }
  
  if (!jobs || jobs.length === 0) {
    return result;
  }
  
  for (const job of jobs as VideoJob[]) {
    result.processed++;
    
    try {
      const engineAdapter = getEngine(job.engine);
      const status = await engineAdapter.checkStatus(job.external_id!);
      
      if (status.status === 'completed' && status.output_url) {
        await supabase
          .from('video_jobs')
          .update({
            status: 'completed',
            output_url: status.output_url,
            completed_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        result.completed++;
        
      } else if (status.status === 'failed') {
        await supabase
          .from('video_jobs')
          .update({
            status: 'failed',
            error_message: status.error || 'Engine returned failed status',
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        result.failed++;
        result.errors.push({ job_id: job.id, error: status.error || 'Failed' });
        
      } else {
        // Still processing - update status if needed
        if (job.status !== 'processing') {
          await supabase
            .from('video_jobs')
            .update({
              status: 'processing',
              updated_at: new Date().toISOString(),
            })
            .eq('id', job.id);
        }
      }
      
    } catch (error: any) {
      result.errors.push({ job_id: job.id, error: error.message });
    }
  }
  
  return result;
}

/**
 * Get queue statistics
 */
export async function getQueueStats(): Promise<{
  by_status: Record<JobStatus, number>;
  by_engine: Record<EngineType, number>;
  oldest_queued: string | null;
  avg_processing_time: number | null;
}> {
  const supabase = getSupabaseAdmin();
  
  const { data: jobs } = await supabase
    .from('video_jobs')
    .select('status, engine, created_at, completed_at');
  
  const stats = {
    by_status: {} as Record<JobStatus, number>,
    by_engine: {} as Record<EngineType, number>,
    oldest_queued: null as string | null,
    avg_processing_time: null as number | null,
  };
  
  if (!jobs) return stats;
  
  // Count by status and engine
  jobs.forEach((job: any) => {
    stats.by_status[job.status] = (stats.by_status[job.status] || 0) + 1;
    stats.by_engine[job.engine] = (stats.by_engine[job.engine] || 0) + 1;
  });
  
  // Find oldest queued job
  const queuedJobs = jobs
    .filter((j: any) => j.status === 'queued')
    .sort((a: any, b: any) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
  
  if (queuedJobs.length > 0) {
    stats.oldest_queued = queuedJobs[0].created_at;
  }
  
  // Calculate average processing time for completed jobs
  const completedJobs = jobs.filter((j: any) => 
    j.status === 'completed' && j.completed_at && j.created_at
  );
  
  if (completedJobs.length > 0) {
    const totalTime = completedJobs.reduce((sum: number, j: any) => {
      const start = new Date(j.created_at).getTime();
      const end = new Date(j.completed_at).getTime();
      return sum + (end - start);
    }, 0);
    stats.avg_processing_time = totalTime / completedJobs.length / 1000; // in seconds
  }
  
  return stats;
}

/**
 * Run full processing cycle
 */
export async function runProcessingCycle(): Promise<{
  queued: ProcessResult;
  pending: ProcessResult;
  stats: ReturnType<typeof getQueueStats> extends Promise<infer T> ? T : never;
}> {
  const queued = await processQueuedJobs();
  const pending = await checkPendingJobs();
  const stats = await getQueueStats();
  
  return { queued, pending, stats };
}
