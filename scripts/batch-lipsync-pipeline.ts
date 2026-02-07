/**
 * Batch Lip-Sync Pipeline (Network-Resilient Edition)
 * 
 * Phase 2: Generate lip-synced videos for all lessons using:
 * - Existing audio from kelly_lesson_assets.audio_url
 * - Base video (Kelly talking video)
 * - Sync Labs lipsync-2-pro API
 * 
 * Features:
 * - Auto-retry with exponential backoff on network failures
 * - Internet connection monitoring with auto-resume
 * - Progress saved to file for crash recovery
 * - Quality validation (file size, accessibility)
 * - Detailed logging to file
 * 
 * Usage:
 *   npx tsx scripts/batch-lipsync-pipeline.ts --test          # Single test (day 34, hook, age35)
 *   npx tsx scripts/batch-lipsync-pipeline.ts --day 34        # Process all assets for day 34
 *   npx tsx scripts/batch-lipsync-pipeline.ts --all           # Process all assets
 *   npx tsx scripts/batch-lipsync-pipeline.ts --status        # Show current status
 *   npx tsx scripts/batch-lipsync-pipeline.ts --retry-failed  # Retry failed assets only
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // Sync Labs API
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  SYNC_LABS_API_URL: 'https://api.sync.so/v2/generate',
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_SERVICE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // Base video for lip-sync
  BASE_VIDEO_URL: 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/064dbfab193c461fbb2869f27d663c7b.mp4',
  
  // Storage
  VIDEO_BUCKET: 'kelly-videos',
  VIDEO_PREFIX: 'lipsync/2026/en',
  
  // Rate limiting & retries
  RATE_LIMIT_MS: 3000,  // 3 seconds between requests
  MAX_RETRIES: 5,       // Increased retries for network issues
  RETRY_BASE_DELAY_MS: 2000,  // Base delay for exponential backoff
  POLL_INTERVAL_MS: 5000,  // Poll job status every 5 seconds
  MAX_POLL_ATTEMPTS: 180,  // 15 minutes max per video
  
  // Network resilience
  NETWORK_CHECK_INTERVAL_MS: 10000,  // Check network every 10s when waiting
  NETWORK_RETRY_DELAY_MS: 30000,     // Wait 30s before retrying after network restore
  
  // Quality validation
  MIN_VIDEO_SIZE_BYTES: 100000,  // 100KB minimum video size
  
  // Logging
  LOG_FILE: 'logs/lipsync-pipeline.log',
  PROGRESS_FILE: 'logs/lipsync-progress.json',
};

interface LessonAsset {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  language: string;
  audio_url: string | null;
  video_url: string | null;
  status: string | null;
}

interface SyncLabsJob {
  id: string;
  status: string;
  output?: { url: string }[];
  outputUrl?: string;
  error?: string;
  message?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

function padDay(day: number): string {
  return day.toString().padStart(3, '0');
}

function getStoragePath(dayNumber: number, phase: string, ageGroup: number): string {
  return `${CONFIG.VIDEO_PREFIX}/day-${padDay(dayNumber)}/${phase}-age${ageGroup}.mp4`;
}

function timestamp(): string {
  return new Date().toISOString().replace('T', ' ').substring(0, 19);
}

function log(emoji: string, message: string, indent: number = 0) {
  const prefix = '  '.repeat(indent);
  const line = `${prefix}${emoji} ${message}`;
  console.log(line);
  appendToLogFile(`[${timestamp()}] ${line}`);
}

function appendToLogFile(message: string) {
  try {
    const logDir = path.dirname(CONFIG.LOG_FILE);
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
    fs.appendFileSync(CONFIG.LOG_FILE, message + '\n');
  } catch (e) {
    // Silently fail if can't write to log
  }
}

// =============================================================================
// NETWORK RESILIENCE
// =============================================================================

async function checkInternetConnection(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch('https://api.sync.so/health', {
      method: 'HEAD',
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    return response.ok || response.status === 404; // 404 is fine, means server responded
  } catch (error) {
    return false;
  }
}

async function waitForInternet(): Promise<void> {
  log('🌐', 'Checking internet connection...');
  
  let attempts = 0;
  while (true) {
    const connected = await checkInternetConnection();
    
    if (connected) {
      if (attempts > 0) {
        log('✅', `Internet restored after ${attempts} checks`);
        // Wait a bit before resuming to let connection stabilize
        log('⏳', `Waiting ${CONFIG.NETWORK_RETRY_DELAY_MS / 1000}s for connection to stabilize...`);
        await sleep(CONFIG.NETWORK_RETRY_DELAY_MS);
      }
      return;
    }
    
    attempts++;
    log('⚠️', `No internet connection (attempt ${attempts}). Waiting ${CONFIG.NETWORK_CHECK_INTERVAL_MS / 1000}s...`);
    await sleep(CONFIG.NETWORK_CHECK_INTERVAL_MS);
  }
}

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = CONFIG.MAX_RETRIES
): Promise<Response> {
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      // Check internet first
      const connected = await checkInternetConnection();
      if (!connected) {
        await waitForInternet();
      }
      
      const response = await fetch(url, options);
      return response;
      
    } catch (error: any) {
      lastError = error;
      const isNetworkError = error.message?.includes('fetch') || 
                            error.message?.includes('network') ||
                            error.message?.includes('ECONNRESET') ||
                            error.message?.includes('ETIMEDOUT') ||
                            error.code === 'ENOTFOUND';
      
      if (isNetworkError) {
        const delay = CONFIG.RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
        log('🔄', `Network error, retry ${attempt + 1}/${maxRetries} in ${delay / 1000}s: ${error.message}`, 2);
        await sleep(delay);
        
        // Check and wait for internet before retrying
        await waitForInternet();
      } else {
        throw error;
      }
    }
  }
  
  throw lastError || new Error('Max retries exceeded');
}

// =============================================================================
// PROGRESS TRACKING
// =============================================================================

interface ProgressData {
  lastUpdated: string;
  totalProcessed: number;
  totalSuccess: number;
  totalFailed: number;
  lastAssetId?: string;
  averageTimePerAssetMs: number;
  startTime: string;
}

function loadProgress(): ProgressData | null {
  try {
    if (fs.existsSync(CONFIG.PROGRESS_FILE)) {
      const data = fs.readFileSync(CONFIG.PROGRESS_FILE, 'utf-8');
      return JSON.parse(data);
    }
  } catch (e) {
    // Ignore errors
  }
  return null;
}

function saveProgress(progress: ProgressData): void {
  try {
    const dir = path.dirname(CONFIG.PROGRESS_FILE);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONFIG.PROGRESS_FILE, JSON.stringify(progress, null, 2));
  } catch (e) {
    // Silently fail
  }
}

function formatETA(remainingAssets: number, avgTimeMs: number): string {
  const totalMs = remainingAssets * avgTimeMs;
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  
  if (hours > 24) {
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    return `~${days}d ${remainingHours}h`;
  }
  return `~${hours}h ${minutes}m`;
}

// =============================================================================
// QUALITY VALIDATION
// =============================================================================

async function validateVideoUrl(url: string): Promise<{ valid: boolean; size?: number; error?: string }> {
  try {
    const response = await fetchWithRetry(url, { method: 'HEAD' }, 3);
    
    if (!response.ok) {
      return { valid: false, error: `HTTP ${response.status}` };
    }
    
    const contentLength = response.headers.get('content-length');
    const size = contentLength ? parseInt(contentLength, 10) : 0;
    
    if (size < CONFIG.MIN_VIDEO_SIZE_BYTES) {
      return { valid: false, size, error: `Video too small: ${size} bytes` };
    }
    
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('video')) {
      return { valid: false, error: `Invalid content type: ${contentType}` };
    }
    
    return { valid: true, size };
    
  } catch (error: any) {
    return { valid: false, error: error.message };
  }
}

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

function getSupabase() {
  if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_SERVICE_KEY) {
    throw new Error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  }
  return createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_SERVICE_KEY);
}

// =============================================================================
// SYNC LABS API (with network resilience)
// =============================================================================

async function submitLipsyncJob(audioUrl: string, videoUrl: string): Promise<string> {
  const response = await fetchWithRetry(CONFIG.SYNC_LABS_API_URL, {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2-pro',
      input: [
        { type: 'video', url: videoUrl },
        { type: 'audio', url: audioUrl },
      ],
    }),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Sync Labs submit error: ${response.status} - ${errorText}`);
  }
  
  const job = await response.json() as SyncLabsJob;
  return job.id;
}

async function pollJobStatus(jobId: string): Promise<{ success: boolean; videoUrl?: string; error?: string }> {
  let consecutiveNetworkErrors = 0;
  
  for (let attempt = 0; attempt < CONFIG.MAX_POLL_ATTEMPTS; attempt++) {
    try {
      const response = await fetchWithRetry(`${CONFIG.SYNC_LABS_API_URL}/${jobId}`, {
        headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
      }, 3);  // 3 retries per poll
      
      consecutiveNetworkErrors = 0;  // Reset on success
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Sync Labs poll error: ${response.status} - ${errorText}`);
      }
      
      const job = await response.json() as SyncLabsJob;
      
      if (job.status === 'COMPLETED') {
        const outputUrl = job.output?.[0]?.url || job.outputUrl;
        if (!outputUrl) {
          return { success: false, error: 'No output URL in completed job' };
        }
        return { success: true, videoUrl: outputUrl };
      }
      
      if (job.status === 'FAILED' || job.status === 'REJECTED') {
        return { success: false, error: job.error || job.message || 'Job failed' };
      }
      
      // Log progress every 12 attempts (1 minute)
      if (attempt % 12 === 0 && attempt > 0) {
        log('⏳', `Still processing... (${Math.round(attempt * CONFIG.POLL_INTERVAL_MS / 1000)}s) [Status: ${job.status}]`, 3);
      }
      
    } catch (error: any) {
      consecutiveNetworkErrors++;
      log('⚠️', `Poll error (${consecutiveNetworkErrors}x): ${error.message}`, 3);
      
      // If we've had too many consecutive network errors, wait for internet
      if (consecutiveNetworkErrors >= 3) {
        await waitForInternet();
        consecutiveNetworkErrors = 0;
      }
    }
    
    await sleep(CONFIG.POLL_INTERVAL_MS);
  }
  
  return { success: false, error: 'Job timed out' };
}

// =============================================================================
// STORAGE UPLOAD (with network resilience & validation)
// =============================================================================

async function downloadAndUpload(
  sourceUrl: string,
  bucket: string,
  storagePath: string
): Promise<string> {
  const supabase = getSupabase();
  
  // Download the video with retry
  log('⬇️', 'Downloading from Sync Labs...', 3);
  const response = await fetchWithRetry(sourceUrl, { method: 'GET' });
  if (!response.ok) {
    throw new Error(`Failed to download video: ${response.status}`);
  }
  
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  
  // Validate downloaded size
  if (bytes.length < CONFIG.MIN_VIDEO_SIZE_BYTES) {
    throw new Error(`Downloaded video too small: ${bytes.length} bytes`);
  }
  
  log('⬆️', `Uploading ${(bytes.length / 1024 / 1024).toFixed(2)}MB to Supabase...`, 3);
  
  // Upload to Supabase Storage with retry
  let uploadAttempt = 0;
  let uploadError: Error | null = null;
  
  while (uploadAttempt < CONFIG.MAX_RETRIES) {
    try {
      const { error } = await supabase.storage
        .from(bucket)
        .upload(storagePath, bytes, {
          contentType: 'video/mp4',
          upsert: true,
        });
      
      if (!error) {
        uploadError = null;
        break;
      }
      
      uploadError = new Error(error.message);
      
    } catch (e: any) {
      uploadError = e;
    }
    
    uploadAttempt++;
    if (uploadAttempt < CONFIG.MAX_RETRIES) {
      const delay = CONFIG.RETRY_BASE_DELAY_MS * Math.pow(2, uploadAttempt);
      log('🔄', `Upload retry ${uploadAttempt}/${CONFIG.MAX_RETRIES} in ${delay / 1000}s...`, 3);
      await sleep(delay);
      await waitForInternet();
    }
  }
  
  if (uploadError) {
    throw new Error(`Upload failed after ${CONFIG.MAX_RETRIES} attempts: ${uploadError.message}`);
  }
  
  // Get public URL
  const { data } = supabase.storage.from(bucket).getPublicUrl(storagePath);
  const publicUrl = data.publicUrl;
  
  // Validate the uploaded video is accessible
  log('🔍', 'Validating uploaded video...', 3);
  const validation = await validateVideoUrl(publicUrl);
  
  if (!validation.valid) {
    throw new Error(`Uploaded video validation failed: ${validation.error}`);
  }
  
  log('✅', `Validated: ${(validation.size! / 1024 / 1024).toFixed(2)}MB`, 3);
  
  return publicUrl;
}

// =============================================================================
// DATABASE OPERATIONS
// =============================================================================

async function getAssetsNeedingVideo(limit?: number, dayFilter?: number): Promise<LessonAsset[]> {
  const supabase = getSupabase();
  
  let query = supabase
    .from('kelly_lesson_assets')
    .select('id, day_number, phase, age_group, language, audio_url, video_url, status')
    .not('audio_url', 'is', null)
    .is('video_url', null)
    .order('day_number')
    .order('phase');
  
  if (dayFilter) {
    query = query.eq('day_number', dayFilter);
  }
  
  if (limit) {
    query = query.limit(limit);
  }
  
  const { data, error } = await query;
  
  if (error) {
    throw new Error(`Database query error: ${error.message}`);
  }
  
  return (data || []) as LessonAsset[];
}

async function getFailedAssets(): Promise<LessonAsset[]> {
  const supabase = getSupabase();
  
  const { data, error } = await supabase
    .from('kelly_lesson_assets')
    .select('id, day_number, phase, age_group, language, audio_url, video_url, status')
    .eq('status', 'failed')
    .not('audio_url', 'is', null)
    .order('day_number')
    .order('phase');
  
  if (error) {
    throw new Error(`Database query error: ${error.message}`);
  }
  
  return (data || []) as LessonAsset[];
}

async function clearFailedStatus(id: string): Promise<void> {
  const supabase = getSupabase();
  
  await supabase
    .from('kelly_lesson_assets')
    .update({
      status: null,
      error_message: null,
      updated_at: new Date().toISOString(),
    })
    .eq('id', id);
}

async function updateAssetVideoUrl(
  id: string,
  videoUrl: string,
  videoSource: string = 'sync_labs_lipsync-2-pro'
): Promise<void> {
  const supabase = getSupabase();
  
  const { error } = await supabase
    .from('kelly_lesson_assets')
    .update({
      video_url: videoUrl,
      video_source: videoSource,
      status: 'complete',
      updated_at: new Date().toISOString(),
    })
    .eq('id', id);
  
  if (error) {
    throw new Error(`Database update error: ${error.message}`);
  }
}

async function markAssetFailed(id: string, errorMessage: string): Promise<void> {
  const supabase = getSupabase();
  
  const { error } = await supabase
    .from('kelly_lesson_assets')
    .update({
      status: 'failed',
      error_message: errorMessage,
      updated_at: new Date().toISOString(),
    })
    .eq('id', id);
  
  if (error) {
    console.error(`Failed to mark asset as failed: ${error.message}`);
  }
}

// =============================================================================
// SINGLE ASSET PROCESSING
// =============================================================================

async function processAsset(asset: LessonAsset): Promise<boolean> {
  const assetId = `Day ${asset.day_number} / ${asset.phase} / age${asset.age_group}`;
  log('🎬', `Processing: ${assetId}`, 1);
  
  if (!asset.audio_url) {
    log('⚠️', 'No audio URL, skipping', 2);
    return false;
  }
  
  try {
    // Step 1: Submit lip-sync job
    log('📤', 'Submitting to Sync Labs...', 2);
    const jobId = await submitLipsyncJob(asset.audio_url, CONFIG.BASE_VIDEO_URL);
    log('✅', `Job ID: ${jobId}`, 2);
    
    // Step 2: Poll for completion
    log('⏳', 'Waiting for completion...', 2);
    const result = await pollJobStatus(jobId);
    
    if (!result.success || !result.videoUrl) {
      throw new Error(result.error || 'Unknown error');
    }
    
    log('✅', 'Lip-sync complete', 2);
    
    // Step 3: Upload to Supabase Storage
    const storagePath = getStoragePath(asset.day_number, asset.phase, asset.age_group);
    log('📤', `Uploading to storage: ${storagePath}`, 2);
    
    const publicUrl = await downloadAndUpload(result.videoUrl, CONFIG.VIDEO_BUCKET, storagePath);
    log('✅', `Uploaded: ${publicUrl}`, 2);
    
    // Step 4: Update database
    await updateAssetVideoUrl(asset.id, publicUrl);
    log('✅', 'Database updated', 2);
    
    return true;
    
  } catch (error: any) {
    log('❌', `Failed: ${error.message}`, 2);
    await markAssetFailed(asset.id, error.message);
    return false;
  }
}

// =============================================================================
// BATCH PROCESSING (with progress tracking & monitoring)
// =============================================================================

async function processBatch(assets: LessonAsset[]): Promise<{ success: number; failed: number }> {
  let success = 0;
  let failed = 0;
  const startTime = Date.now();
  const processingTimes: number[] = [];
  
  // Load existing progress
  let progress = loadProgress() || {
    lastUpdated: new Date().toISOString(),
    totalProcessed: 0,
    totalSuccess: 0,
    totalFailed: 0,
    averageTimePerAssetMs: 300000, // Default 5 min estimate
    startTime: new Date().toISOString(),
  };
  
  for (let i = 0; i < assets.length; i++) {
    const asset = assets[i];
    const assetStartTime = Date.now();
    
    // Calculate ETA
    const avgTime = processingTimes.length > 0 
      ? processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length 
      : progress.averageTimePerAssetMs;
    const remaining = assets.length - i;
    const eta = formatETA(remaining, avgTime);
    
    console.log(''); // Blank line for readability
    log('📊', `Progress: ${i + 1}/${assets.length} | Success: ${success} | Failed: ${failed} | ETA: ${eta}`, 0);
    
    const result = await processAsset(asset);
    const assetTime = Date.now() - assetStartTime;
    processingTimes.push(assetTime);
    
    // Keep only last 10 times for rolling average
    if (processingTimes.length > 10) {
      processingTimes.shift();
    }
    
    if (result) {
      success++;
      log('⏱️', `Completed in ${Math.round(assetTime / 1000)}s`, 2);
    } else {
      failed++;
    }
    
    // Update and save progress
    progress = {
      lastUpdated: new Date().toISOString(),
      totalProcessed: i + 1,
      totalSuccess: success,
      totalFailed: failed,
      lastAssetId: asset.id,
      averageTimePerAssetMs: avgTime,
      startTime: progress.startTime,
    };
    saveProgress(progress);
    
    // Log summary every 10 assets
    if ((i + 1) % 10 === 0) {
      const elapsed = Math.round((Date.now() - startTime) / 60000);
      log('📈', `Checkpoint: ${success} success, ${failed} failed in ${elapsed} minutes`, 0);
    }
    
    // Rate limiting between requests
    if (i < assets.length - 1) {
      log('⏱️', `Rate limit pause (${CONFIG.RATE_LIMIT_MS / 1000}s)...`, 1);
      await sleep(CONFIG.RATE_LIMIT_MS);
    }
  }
  
  return { success, failed };
}

// =============================================================================
// STATUS REPORT (enhanced)
// =============================================================================

async function showStatus(): Promise<void> {
  const supabase = getSupabase();
  
  console.log('\n' + '='.repeat(60));
  console.log('  Kelly Lip-Sync Pipeline Status');
  console.log('='.repeat(60) + '\n');
  
  // Show progress file info
  const progress = loadProgress();
  if (progress) {
    console.log('  📊 Last Run:');
    console.log(`    Started:     ${progress.startTime}`);
    console.log(`    Last update: ${progress.lastUpdated}`);
    console.log(`    Processed:   ${progress.totalProcessed}`);
    console.log(`    Success:     ${progress.totalSuccess}`);
    console.log(`    Failed:      ${progress.totalFailed}`);
    console.log(`    Avg time:    ${Math.round(progress.averageTimePerAssetMs / 1000)}s per video`);
    console.log('');
  }
  
  // Check network
  console.log('  🌐 Network Status:');
  const connected = await checkInternetConnection();
  console.log(`    Internet: ${connected ? '✅ Connected' : '❌ Disconnected'}`);
  console.log('');
  
  // Total counts using count queries (more efficient)
  const { count: totalCount } = await supabase
    .from('kelly_lesson_assets')
    .select('*', { count: 'exact', head: true });
  
  const { count: withAudio } = await supabase
    .from('kelly_lesson_assets')
    .select('*', { count: 'exact', head: true })
    .not('audio_url', 'is', null);
  
  const { count: withVideo } = await supabase
    .from('kelly_lesson_assets')
    .select('*', { count: 'exact', head: true })
    .not('video_url', 'is', null);
  
  const { count: needingVideo } = await supabase
    .from('kelly_lesson_assets')
    .select('*', { count: 'exact', head: true })
    .not('audio_url', 'is', null)
    .is('video_url', null);
  
  const { count: failedCount } = await supabase
    .from('kelly_lesson_assets')
    .select('*', { count: 'exact', head: true })
    .eq('status', 'failed');
  
  const total = totalCount || 0;
  const audio = withAudio || 0;
  const video = withVideo || 0;
  const needing = needingVideo || 0;
  const failedNum = failedCount || 0;
  
  const pctComplete = audio > 0 ? Math.round((video / audio) * 100) : 0;
  
  console.log('  📦 Asset Counts:');
  console.log(`    Total assets:        ${total}`);
  console.log(`    With audio:          ${audio}`);
  console.log(`    With video:          ${video} (${pctComplete}% complete)`);
  console.log(`    Needing lip-sync:    ${needing}`);
  console.log(`    Failed:              ${failedNum}`);
  
  // ETA calculation
  if (needing > 0 && progress) {
    const eta = formatETA(needing, progress.averageTimePerAssetMs);
    console.log(`\n  ⏱️  Estimated time remaining: ${eta}`);
  }
  
  // Per-day breakdown
  console.log('\n  📅 Per-Day Status:');
  
  // Get counts grouped by day
  const { data: dayCounts } = await supabase
    .from('kelly_lesson_assets')
    .select('day_number')
    .not('video_url', 'is', null);
  
  const dayVideoMap = new Map<number, number>();
  (dayCounts || []).forEach((row: any) => {
    const current = dayVideoMap.get(row.day_number) || 0;
    dayVideoMap.set(row.day_number, current + 1);
  });
  
  // Show first 15 days and last 5 days
  const daysToShow = [...Array(15).keys()].map(i => i + 1);
  
  for (const day of daysToShow) {
    const { count: dayTotal } = await supabase
      .from('kelly_lesson_assets')
      .select('*', { count: 'exact', head: true })
      .eq('day_number', day);
    
    const doneCount = dayVideoMap.get(day) || 0;
    const totalDay = dayTotal || 0;
    const pct = totalDay > 0 ? Math.round((doneCount / totalDay) * 100) : 0;
    const bar = '█'.repeat(Math.floor(pct / 10)) + '░'.repeat(10 - Math.floor(pct / 10));
    
    console.log(`    Day ${day.toString().padStart(3, '0')}: ${bar} ${doneCount}/${totalDay} (${pct}%)`);
  }
  
  console.log('    ...');
  
  // Recent activity (last 5 completed)
  const { data: recent } = await supabase
    .from('kelly_lesson_assets')
    .select('day_number, phase, age_group, updated_at')
    .not('video_url', 'is', null)
    .order('updated_at', { ascending: false })
    .limit(5);
  
  if (recent && recent.length > 0) {
    console.log('\n  🕐 Recently Completed:');
    for (const r of recent) {
      console.log(`    Day ${r.day_number} / ${r.phase} / age${r.age_group} @ ${r.updated_at}`);
    }
  }
  
  console.log('\n' + '='.repeat(60) + '\n');
}

// =============================================================================
// MAIN (with enhanced startup checks)
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  
  // Validate configuration
  if (!CONFIG.SYNC_LABS_API_KEY) {
    console.error('❌ Missing SYNC_LABS_API_KEY in environment');
    process.exit(1);
  }
  
  if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_SERVICE_KEY) {
    console.error('❌ Missing Supabase configuration');
    process.exit(1);
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('  Kelly Lip-Sync Batch Pipeline (Network-Resilient)');
  console.log('='.repeat(60));
  console.log(`\n  Started: ${timestamp()}`);
  console.log(`  Sync Labs API: ${CONFIG.SYNC_LABS_API_KEY ? '✅ Configured' : '❌ Missing'}`);
  console.log(`  Log file: ${CONFIG.LOG_FILE}`);
  console.log(`  Progress file: ${CONFIG.PROGRESS_FILE}`);
  console.log('');
  
  // Initial network check
  log('🌐', 'Checking internet connection...');
  const connected = await checkInternetConnection();
  if (!connected) {
    log('⚠️', 'No internet connection detected. Will retry...');
    await waitForInternet();
  }
  log('✅', 'Internet connection OK');
  
  // Parse arguments
  if (args.includes('--status')) {
    await showStatus();
    return;
  }
  
  if (args.includes('--test')) {
    // Test mode: Single asset (day 34, hook, age35)
    console.log('  Mode: SINGLE ASSET TEST\n');
    
    const supabase = getSupabase();
    const { data: asset, error } = await supabase
      .from('kelly_lesson_assets')
      .select('*')
      .eq('day_number', 34)
      .eq('phase', 'hook')
      .eq('age_group', 35)
      .single();
    
    if (error || !asset) {
      console.error('❌ Could not find test asset (day 34, hook, age35)');
      process.exit(1);
    }
    
    const result = await processAsset(asset as LessonAsset);
    
    console.log('\n' + '='.repeat(60));
    console.log(result ? '  ✅ TEST PASSED' : '  ❌ TEST FAILED');
    console.log('='.repeat(60) + '\n');
    
    process.exit(result ? 0 : 1);
  }
  
  if (args.includes('--retry-failed')) {
    // Retry failed assets
    console.log('  Mode: RETRY FAILED ASSETS\n');
    
    const failedAssets = await getFailedAssets();
    
    if (failedAssets.length === 0) {
      console.log('  ✅ No failed assets to retry');
      process.exit(0);
    }
    
    console.log(`  Found ${failedAssets.length} failed assets to retry\n`);
    
    // Clear failed status before retrying
    for (const asset of failedAssets) {
      await clearFailedStatus(asset.id);
    }
    
    const startTime = Date.now();
    const result = await processBatch(failedAssets);
    const duration = Math.round((Date.now() - startTime) / 1000);
    
    console.log('\n' + '='.repeat(60));
    console.log('  RETRY COMPLETE');
    console.log('='.repeat(60));
    console.log(`\n  Success: ${result.success}`);
    console.log(`  Still failed: ${result.failed}`);
    console.log(`  Duration: ${duration}s`);
    console.log('');
    
    process.exit(result.failed > 0 ? 1 : 0);
  }
  
  const dayIndex = args.indexOf('--day');
  const dayFilter = dayIndex >= 0 ? parseInt(args[dayIndex + 1], 10) : undefined;
  
  if (dayFilter) {
    console.log(`  Mode: PROCESS DAY ${dayFilter}\n`);
  } else if (args.includes('--all')) {
    console.log('  Mode: PROCESS ALL\n');
    
    // Show warning for large batch
    const progress = loadProgress();
    if (progress) {
      console.log(`  Previous run: ${progress.totalProcessed} processed, ${progress.totalSuccess} success`);
      console.log(`  Last updated: ${progress.lastUpdated}`);
      console.log('');
    }
  } else {
    console.log('Usage:');
    console.log('  npx tsx scripts/batch-lipsync-pipeline.ts --test          # Test single asset');
    console.log('  npx tsx scripts/batch-lipsync-pipeline.ts --day 34        # Process day 34');
    console.log('  npx tsx scripts/batch-lipsync-pipeline.ts --all           # Process all');
    console.log('  npx tsx scripts/batch-lipsync-pipeline.ts --status        # Show status');
    console.log('  npx tsx scripts/batch-lipsync-pipeline.ts --retry-failed  # Retry failed only');
    process.exit(0);
  }
  
  // Get assets needing processing
  const assets = await getAssetsNeedingVideo(undefined, dayFilter);
  
  if (assets.length === 0) {
    console.log('  ✅ No assets need lip-sync processing');
    process.exit(0);
  }
  
  // Calculate estimates
  const progress = loadProgress();
  const avgTime = progress?.averageTimePerAssetMs || 300000;
  const eta = formatETA(assets.length, avgTime);
  
  console.log(`  Found ${assets.length} assets needing lip-sync`);
  console.log(`  Estimated time: ${eta}`);
  console.log('');
  
  // Process batch
  const startTime = Date.now();
  const result = await processBatch(assets);
  const duration = Math.round((Date.now() - startTime) / 1000);
  const hours = Math.floor(duration / 3600);
  const minutes = Math.floor((duration % 3600) / 60);
  
  console.log('\n' + '='.repeat(60));
  console.log('  BATCH COMPLETE');
  console.log('='.repeat(60));
  console.log(`\n  Finished: ${timestamp()}`);
  console.log(`  Success: ${result.success}`);
  console.log(`  Failed:  ${result.failed}`);
  console.log(`  Duration: ${hours}h ${minutes}m ${duration % 60}s`);
  console.log(`  Log file: ${CONFIG.LOG_FILE}`);
  console.log('');
  
  // Log completion
  appendToLogFile(`\n=== BATCH COMPLETE ===`);
  appendToLogFile(`Success: ${result.success}, Failed: ${result.failed}, Duration: ${duration}s`);
  
  process.exit(result.failed > 0 ? 1 : 0);
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n⚠️ Interrupted! Progress has been saved.');
  console.log(`   Check ${CONFIG.PROGRESS_FILE} for status.`);
  console.log('   Run the same command again to resume.\n');
  process.exit(130);
});

process.on('SIGTERM', () => {
  console.log('\n\n⚠️ Terminated! Progress has been saved.');
  process.exit(143);
});

main().catch((error) => {
  console.error('Fatal error:', error);
  appendToLogFile(`FATAL ERROR: ${error.message}`);
  process.exit(1);
});
