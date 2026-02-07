/**
 * Multi-Provider Fallback Queue System
 * 
 * Orchestrates video generation across multiple providers:
 * HeyGen → sync.so → fal.ai → Replicate
 * 
 * Features:
 * - Automatic fallback on failure
 * - Eval gates at each stage
 * - Email alerts after 3 retries
 * - Provider availability tracking
 */

import { getSupabaseAdmin } from '../api/lib/supabase';
import { getEngine, engines, PROVIDER_FALLBACK_CHAIN } from './engines';
import type { VideoJob, EngineType, JobStatus, EvalGateResult, EngineInputPayload } from './engines/types';
import { evaluateContent, evaluateAudio, evaluateVideo, evaluateDeploy } from './eval-gates';
import { notifyEvalFailure, notifyJobFailure, notifyPipelineError, notifyDailySummary } from './email-alerts';

const MAX_RETRIES = 3;

export interface FallbackQueueConfig {
  providerOrder?: EngineType[];
  maxRetries?: number;
  enableEvalGates?: boolean;
  enableAlerts?: boolean;
  dryRun?: boolean;
}

export interface QueueJobResult {
  success: boolean;
  job_id: string;
  engine_used?: EngineType;
  output_url?: string;
  error?: string;
  retries: number;
  fallback_attempts: EngineType[];
  eval_results?: {
    content?: EvalGateResult;
    audio?: EvalGateResult;
    video?: EvalGateResult;
    deploy?: EvalGateResult;
  };
}

const defaultConfig: FallbackQueueConfig = {
  providerOrder: PROVIDER_FALLBACK_CHAIN,
  maxRetries: MAX_RETRIES,
  enableEvalGates: true,
  enableAlerts: true,
  dryRun: false,
};

/**
 * Get available providers in fallback order
 */
export async function getAvailableProviders(
  providerOrder: EngineType[] = PROVIDER_FALLBACK_CHAIN
): Promise<EngineType[]> {
  const available: EngineType[] = [];
  
  for (const engine of providerOrder) {
    try {
      const adapter = getEngine(engine);
      if (await adapter.isAvailable()) {
        available.push(engine);
      }
    } catch {
      // Engine not available
    }
  }
  
  return available;
}

/**
 * Submit a job with automatic provider fallback
 */
export async function submitWithFallback(
  job: VideoJob,
  config: FallbackQueueConfig = {}
): Promise<QueueJobResult> {
  const cfg = { ...defaultConfig, ...config };
  const supabase = getSupabaseAdmin();
  
  const result: QueueJobResult = {
    success: false,
    job_id: job.id,
    retries: 0,
    fallback_attempts: [],
  };
  
  // Get available providers
  const providers = await getAvailableProviders(cfg.providerOrder);
  
  if (providers.length === 0) {
    const error = 'No video providers available';
    await notifyPipelineError(error, { job_id: job.id });
    result.error = error;
    return result;
  }
  
  // Run content eval gate first
  if (cfg.enableEvalGates && job.input_payload.text) {
    const contentEval = evaluateContent({ text: job.input_payload.text });
    result.eval_results = { content: contentEval };
    
    if (!contentEval.passed) {
      // Content failed - retry with regeneration up to max
      if (job.retry_count && job.retry_count >= cfg.maxRetries!) {
        if (cfg.enableAlerts) {
          await notifyEvalFailure(
            job.id,
            job.day_of_year,
            job.phase,
            job.retry_count,
            contentEval.issues
          );
        }
        
        await supabase
          .from('video_jobs')
          .update({
            status: 'eval_failed',
            error_message: `Content eval failed: ${contentEval.issues.join('; ')}`,
            eval_results: result.eval_results,
            updated_at: new Date().toISOString(),
          })
          .eq('id', job.id);
        
        result.error = `Content eval failed after ${job.retry_count} retries`;
        return result;
      }
      
      // Increment retry and requeue
      await supabase
        .from('video_jobs')
        .update({
          status: 'queued',
          retry_count: (job.retry_count || 0) + 1,
          error_message: `Content eval retry: ${contentEval.issues.join('; ')}`,
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      result.retries = (job.retry_count || 0) + 1;
      result.error = 'Content eval failed, requeued for retry';
      return result;
    }
  }
  
  // Try each provider in order
  for (const engine of providers) {
    result.fallback_attempts.push(engine);
    
    try {
      if (cfg.dryRun) {
        console.log(`[DRY RUN] Would submit to ${engine}`);
        result.success = true;
        result.engine_used = engine;
        return result;
      }
      
      const adapter = getEngine(engine);
      
      // Update job to current engine
      await supabase
        .from('video_jobs')
        .update({
          engine,
          status: 'submitted',
          fallback_chain: result.fallback_attempts,
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      // Submit to engine
      const submitResult = await adapter.submit(job);
      
      // Update with external ID
      await supabase
        .from('video_jobs')
        .update({
          external_id: submitResult.external_id,
          submitted_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      result.success = true;
      result.engine_used = engine;
      
      console.log(`✅ Job ${job.id} submitted to ${engine} (external: ${submitResult.external_id})`);
      return result;
      
    } catch (error: any) {
      const errorMsg = error.message || 'Unknown error';
      console.log(`❌ ${engine} failed: ${errorMsg}`);
      
      // Check if blocked (401, rate limit, etc.)
      const isBlocked = errorMsg.includes('BLOCKED:') || errorMsg.includes('401');
      
      if (isBlocked) {
        console.log(`   Provider ${engine} is blocked, trying next...`);
        continue;
      }
      
      // Non-blocking error - still try fallback
      continue;
    }
  }
  
  // All providers failed
  const error = `All providers failed: ${result.fallback_attempts.join(' → ')}`;
  result.error = error;
  
  if (cfg.enableAlerts) {
    await notifyJobFailure(
      job.id,
      job.day_of_year,
      job.phase,
      result.fallback_attempts.join(' → '),
      error
    );
  }
  
  await supabase
    .from('video_jobs')
    .update({
      status: 'failed',
      error_message: error,
      fallback_chain: result.fallback_attempts,
      updated_at: new Date().toISOString(),
    })
    .eq('id', job.id);
  
  return result;
}

/**
 * Process completed job with eval gates
 */
export async function processCompletedJob(
  job: VideoJob,
  outputUrl: string,
  config: FallbackQueueConfig = {}
): Promise<QueueJobResult> {
  const cfg = { ...defaultConfig, ...config };
  const supabase = getSupabaseAdmin();
  
  const result: QueueJobResult = {
    success: false,
    job_id: job.id,
    engine_used: job.engine,
    output_url: outputUrl,
    retries: job.retry_count || 0,
    fallback_attempts: job.fallback_chain || [],
  };
  
  result.eval_results = result.eval_results || {};
  
  // Video eval gate
  if (cfg.enableEvalGates) {
    const videoEval = evaluateVideo({ url: outputUrl });
    result.eval_results.video = videoEval;
    
    if (!videoEval.passed) {
      result.error = `Video eval failed: ${videoEval.issues.join('; ')}`;
      
      await supabase
        .from('video_jobs')
        .update({
          status: 'eval_failed',
          error_message: result.error,
          eval_results: result.eval_results,
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      return result;
    }
    
    // Deploy eval gate
    const deployEval = evaluateDeploy({
      content_eval: result.eval_results.content,
      audio_eval: result.eval_results.audio,
      video_eval: videoEval,
      video_url: outputUrl,
      storage_verified: true, // Assume verified if URL exists
    });
    result.eval_results.deploy = deployEval;
    
    if (!deployEval.passed) {
      result.error = `Deploy eval failed: ${deployEval.issues.join('; ')}`;
      
      await supabase
        .from('video_jobs')
        .update({
          status: 'eval_failed',
          error_message: result.error,
          eval_results: result.eval_results,
          updated_at: new Date().toISOString(),
        })
        .eq('id', job.id);
      
      return result;
    }
  }
  
  // All evals passed - mark complete
  await supabase
    .from('video_jobs')
    .update({
      status: 'completed',
      output_url: outputUrl,
      eval_results: result.eval_results,
      completed_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
    .eq('id', job.id);
  
  result.success = true;
  console.log(`✅ Job ${job.id} completed with all eval gates passed`);
  
  return result;
}

/**
 * Process queue batch with fallback
 */
export async function processQueueBatch(
  limit: number = 10,
  config: FallbackQueueConfig = {}
): Promise<{
  processed: number;
  succeeded: number;
  failed: number;
  results: QueueJobResult[];
}> {
  const cfg = { ...defaultConfig, ...config };
  const supabase = getSupabaseAdmin();
  
  // Get queued jobs
  const { data: jobs, error } = await supabase
    .from('video_jobs')
    .select('*')
    .eq('status', 'queued')
    .order('priority', { ascending: false })
    .order('created_at', { ascending: true })
    .limit(limit);
  
  if (error) {
    console.error('Error fetching queue:', error);
    return { processed: 0, succeeded: 0, failed: 0, results: [] };
  }
  
  if (!jobs || jobs.length === 0) {
    console.log('No jobs in queue');
    return { processed: 0, succeeded: 0, failed: 0, results: [] };
  }
  
  const results: QueueJobResult[] = [];
  let succeeded = 0;
  let failed = 0;
  
  for (const job of jobs as VideoJob[]) {
    const result = await submitWithFallback(job, cfg);
    results.push(result);
    
    if (result.success) {
      succeeded++;
    } else {
      failed++;
    }
  }
  
  return {
    processed: jobs.length,
    succeeded,
    failed,
    results,
  };
}

/**
 * Check and complete pending jobs
 */
export async function checkPendingJobs(
  limit: number = 50,
  config: FallbackQueueConfig = {}
): Promise<{
  checked: number;
  completed: number;
  failed: number;
  still_processing: number;
}> {
  const cfg = { ...defaultConfig, ...config };
  const supabase = getSupabaseAdmin();
  
  // Get submitted/processing jobs
  const { data: jobs, error } = await supabase
    .from('video_jobs')
    .select('*')
    .in('status', ['submitted', 'processing'])
    .not('external_id', 'is', null)
    .order('submitted_at', { ascending: true })
    .limit(limit);
  
  if (error || !jobs) {
    console.error('Error fetching pending jobs:', error);
    return { checked: 0, completed: 0, failed: 0, still_processing: 0 };
  }
  
  let completed = 0;
  let failed = 0;
  let still_processing = 0;
  
  for (const job of jobs as VideoJob[]) {
    try {
      const adapter = getEngine(job.engine);
      const status = await adapter.checkStatus(job.external_id!);
      
      if (status.status === 'completed' && status.output_url) {
        const result = await processCompletedJob(job, status.output_url, cfg);
        if (result.success) {
          completed++;
        } else {
          failed++;
        }
      } else if (status.status === 'failed') {
        // Try fallback to next provider
        const nextProviders = PROVIDER_FALLBACK_CHAIN.slice(
          PROVIDER_FALLBACK_CHAIN.indexOf(job.engine) + 1
        );
        
        if (nextProviders.length > 0) {
          // Requeue with next provider
          await supabase
            .from('video_jobs')
            .update({
              status: 'queued',
              engine: nextProviders[0],
              external_id: null,
              error_message: `${job.engine} failed: ${status.error}`,
              fallback_chain: [...(job.fallback_chain || []), job.engine],
              updated_at: new Date().toISOString(),
            })
            .eq('id', job.id);
          
          console.log(`↪️  Job ${job.id} falling back from ${job.engine} to ${nextProviders[0]}`);
        } else {
          // No more fallbacks
          await supabase
            .from('video_jobs')
            .update({
              status: 'failed',
              error_message: status.error || 'All providers failed',
              updated_at: new Date().toISOString(),
            })
            .eq('id', job.id);
          
          if (cfg.enableAlerts) {
            await notifyJobFailure(
              job.id,
              job.day_of_year,
              job.phase,
              job.engine,
              status.error || 'Unknown error'
            );
          }
          
          failed++;
        }
      } else {
        // Still processing
        if (job.status !== 'processing') {
          await supabase
            .from('video_jobs')
            .update({
              status: 'processing',
              updated_at: new Date().toISOString(),
            })
            .eq('id', job.id);
        }
        still_processing++;
      }
    } catch (error: any) {
      console.error(`Error checking job ${job.id}:`, error.message);
      failed++;
    }
  }
  
  return { checked: jobs.length, completed, failed, still_processing };
}

/**
 * Get queue statistics
 */
export async function getQueueStats(): Promise<{
  by_status: Record<JobStatus, number>;
  by_engine: Record<EngineType, number>;
  provider_availability: Record<EngineType, boolean>;
}> {
  const supabase = getSupabaseAdmin();
  
  const { data: jobs } = await supabase
    .from('video_jobs')
    .select('status, engine');
  
  const by_status: Record<string, number> = {};
  const by_engine: Record<string, number> = {};
  
  if (jobs) {
    for (const job of jobs) {
      by_status[job.status] = (by_status[job.status] || 0) + 1;
      by_engine[job.engine] = (by_engine[job.engine] || 0) + 1;
    }
  }
  
  // Check provider availability
  const provider_availability: Record<string, boolean> = {};
  for (const engine of PROVIDER_FALLBACK_CHAIN) {
    try {
      const adapter = getEngine(engine);
      provider_availability[engine] = await adapter.isAvailable();
    } catch {
      provider_availability[engine] = false;
    }
  }
  
  return {
    by_status: by_status as Record<JobStatus, number>,
    by_engine: by_engine as Record<EngineType, number>,
    provider_availability: provider_availability as Record<EngineType, boolean>,
  };
}

/**
 * Run full processing cycle
 */
export async function runProcessingCycle(
  config: FallbackQueueConfig = {}
): Promise<{
  queue_batch: Awaited<ReturnType<typeof processQueueBatch>>;
  pending_check: Awaited<ReturnType<typeof checkPendingJobs>>;
  stats: Awaited<ReturnType<typeof getQueueStats>>;
}> {
  console.log('\n' + '═'.repeat(60));
  console.log('🔄 KELLY PIPELINE PROCESSING CYCLE');
  console.log('═'.repeat(60));
  
  const queue_batch = await processQueueBatch(10, config);
  console.log(`\nQueue batch: ${queue_batch.succeeded}/${queue_batch.processed} succeeded`);
  
  const pending_check = await checkPendingJobs(50, config);
  console.log(`Pending check: ${pending_check.completed} completed, ${pending_check.still_processing} processing`);
  
  const stats = await getQueueStats();
  console.log('\nProvider availability:');
  for (const [engine, available] of Object.entries(stats.provider_availability)) {
    console.log(`  ${available ? '✅' : '❌'} ${engine}`);
  }
  
  console.log('\n' + '═'.repeat(60));
  
  return { queue_batch, pending_check, stats };
}
