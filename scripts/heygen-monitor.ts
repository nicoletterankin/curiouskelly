#!/usr/bin/env npx tsx
/**
 * HEYGEN PERFORMANCE MONITOR
 * 
 * Monitors HeyGen video generation performance and keeps records.
 * Tracks queue times, completion rates, and processing speeds.
 * 
 * Usage:
 *   npx tsx scripts/heygen-monitor.ts                    # Check all queued videos
 *   npx tsx scripts/heygen-monitor.ts --report           # Generate performance report
 *   npx tsx scripts/heygen-monitor.ts --watch            # Watch mode (check every 5 min)
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const LOGS_DIR = path.join(process.cwd(), 'logs', 'heygen');
const PERFORMANCE_LOG = path.join(LOGS_DIR, 'performance.json');

interface VideoRecord {
  video_id: string;
  day: number;
  year: number;
  submitted_at: string;
  first_seen_processing?: string;
  completed_at?: string;
  status: string;
  video_url?: string;
  duration?: number;
  queue_time_minutes?: number;
  processing_time_minutes?: number;
  total_time_minutes?: number;
}

interface PerformanceLog {
  last_updated: string;
  videos: { [key: string]: VideoRecord };
  stats: {
    total_submitted: number;
    total_completed: number;
    total_failed: number;
    total_pending: number;
    avg_queue_time_minutes: number;
    avg_processing_time_minutes: number;
    avg_total_time_minutes: number;
  };
}

// ═══════════════════════════════════════════════════════════════════
// FILE OPERATIONS
// ═══════════════════════════════════════════════════════════════════

function ensureLogsDir(): void {
  if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true });
  }
}

function loadPerformanceLog(): PerformanceLog {
  ensureLogsDir();
  
  if (fs.existsSync(PERFORMANCE_LOG)) {
    return JSON.parse(fs.readFileSync(PERFORMANCE_LOG, 'utf-8'));
  }
  
  return {
    last_updated: new Date().toISOString(),
    videos: {},
    stats: {
      total_submitted: 0,
      total_completed: 0,
      total_failed: 0,
      total_pending: 0,
      avg_queue_time_minutes: 0,
      avg_processing_time_minutes: 0,
      avg_total_time_minutes: 0
    }
  };
}

function savePerformanceLog(log: PerformanceLog): void {
  ensureLogsDir();
  log.last_updated = new Date().toISOString();
  fs.writeFileSync(PERFORMANCE_LOG, JSON.stringify(log, null, 2));
}

function appendToCheckLog(message: string): void {
  ensureLogsDir();
  const logFile = path.join(LOGS_DIR, `checks-${new Date().toISOString().split('T')[0]}.log`);
  const timestamp = new Date().toISOString();
  fs.appendFileSync(logFile, `[${timestamp}] ${message}\n`);
}

// ═══════════════════════════════════════════════════════════════════
// QUEUE FILE LOADING
// ═══════════════════════════════════════════════════════════════════

interface QueueFile {
  month: string;
  year: number;
  videos: { [day: string]: string };
}

function loadAllQueues(): { video_id: string; day: number; year: number; month: string }[] {
  const queueDir = path.join(process.cwd(), 'content', 'email-summary-video');
  const queues: { video_id: string; day: number; year: number; month: string }[] = [];
  
  const months = [
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december'
  ];
  
  for (const month of months) {
    const queueFile = path.join(queueDir, `${month}-video-queue.json`);
    if (fs.existsSync(queueFile)) {
      const data = JSON.parse(fs.readFileSync(queueFile, 'utf-8'));
      for (const [day, videoId] of Object.entries(data.videos || {})) {
        queues.push({
          video_id: videoId as string,
          day: parseInt(day),
          year: 1,
          month
        });
      }
    }
  }
  
  return queues;
}

// ═══════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

async function checkVideoStatus(videoId: string): Promise<{
  status: string;
  video_url?: string;
  duration?: number;
  error?: string;
}> {
  try {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`,
      { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
    );
    
    const result = await response.json();
    
    return {
      status: result.data?.status || 'unknown',
      video_url: result.data?.video_url,
      duration: result.data?.duration,
      error: result.data?.error
    };
  } catch (err) {
    return { status: 'error', error: String(err) };
  }
}

// ═══════════════════════════════════════════════════════════════════
// MONITORING
// ═══════════════════════════════════════════════════════════════════

async function checkAllVideos(): Promise<void> {
  console.log('\n🔍 Loading video queues...');
  const allVideos = loadAllQueues();
  console.log(`   Found ${allVideos.length} videos to check\n`);
  
  const log = loadPerformanceLog();
  const now = new Date();
  
  let completed = 0;
  let pending = 0;
  let failed = 0;
  let newCompletions = 0;
  
  // Check in batches to avoid rate limiting
  const batchSize = 10;
  for (let i = 0; i < allVideos.length; i += batchSize) {
    const batch = allVideos.slice(i, i + batchSize);
    
    const results = await Promise.all(
      batch.map(async (v) => {
        const status = await checkVideoStatus(v.video_id);
        return { ...v, ...status };
      })
    );
    
    for (const result of results) {
      const key = `y${result.year}-d${result.day}`;
      const existing = log.videos[key];
      
      // Initialize or update record
      if (!existing) {
        log.videos[key] = {
          video_id: result.video_id,
          day: result.day,
          year: result.year,
          submitted_at: now.toISOString(),
          status: result.status
        };
      }
      
      const record = log.videos[key];
      
      // Track status transitions
      if (result.status === 'processing' && !record.first_seen_processing) {
        record.first_seen_processing = now.toISOString();
        record.queue_time_minutes = Math.round(
          (now.getTime() - new Date(record.submitted_at).getTime()) / 60000
        );
      }
      
      if (result.status === 'completed' && record.status !== 'completed') {
        record.completed_at = now.toISOString();
        record.video_url = result.video_url;
        record.duration = result.duration;
        
        const submitted = new Date(record.submitted_at);
        record.total_time_minutes = Math.round((now.getTime() - submitted.getTime()) / 60000);
        
        if (record.first_seen_processing) {
          const processing = new Date(record.first_seen_processing);
          record.processing_time_minutes = Math.round((now.getTime() - processing.getTime()) / 60000);
        }
        
        newCompletions++;
        appendToCheckLog(`COMPLETED: Day ${result.day} Year ${result.year} - ${result.video_url}`);
      }
      
      record.status = result.status;
      
      // Count stats
      if (result.status === 'completed') completed++;
      else if (result.status === 'failed') failed++;
      else pending++;
    }
    
    // Progress indicator
    process.stdout.write(`   Checked ${Math.min(i + batchSize, allVideos.length)}/${allVideos.length}\r`);
    
    // Small delay between batches
    if (i + batchSize < allVideos.length) {
      await new Promise(r => setTimeout(r, 200));
    }
  }
  
  // Calculate stats
  const completedRecords = Object.values(log.videos).filter(v => v.status === 'completed');
  
  log.stats = {
    total_submitted: allVideos.length,
    total_completed: completed,
    total_failed: failed,
    total_pending: pending,
    avg_queue_time_minutes: completedRecords.length > 0
      ? Math.round(completedRecords.reduce((sum, v) => sum + (v.queue_time_minutes || 0), 0) / completedRecords.length)
      : 0,
    avg_processing_time_minutes: completedRecords.length > 0
      ? Math.round(completedRecords.reduce((sum, v) => sum + (v.processing_time_minutes || 0), 0) / completedRecords.length)
      : 0,
    avg_total_time_minutes: completedRecords.length > 0
      ? Math.round(completedRecords.reduce((sum, v) => sum + (v.total_time_minutes || 0), 0) / completedRecords.length)
      : 0
  };
  
  savePerformanceLog(log);
  appendToCheckLog(`CHECK: ${completed} completed, ${pending} pending, ${failed} failed`);
  
  console.log('\n');
  console.log('════════════════════════════════════════════════════════════════');
  console.log('📊 HEYGEN PERFORMANCE SUMMARY');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`   Last Updated: ${log.last_updated}`);
  console.log('');
  console.log(`   ✅ Completed: ${completed}`);
  console.log(`   ⏸️  Pending:   ${pending}`);
  console.log(`   ❌ Failed:    ${failed}`);
  console.log('');
  
  if (newCompletions > 0) {
    console.log(`   🎉 NEW COMPLETIONS: ${newCompletions}`);
  }
  
  if (completedRecords.length > 0) {
    console.log('   📈 Average Times (completed videos):');
    console.log(`      Queue time:      ${log.stats.avg_queue_time_minutes} min`);
    console.log(`      Processing time: ${log.stats.avg_processing_time_minutes} min`);
    console.log(`      Total time:      ${log.stats.avg_total_time_minutes} min`);
  }
  
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`   Log saved to: ${PERFORMANCE_LOG}`);
}

async function watchMode(): Promise<void> {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  👁️  HEYGEN WATCH MODE                                          ║');
  console.log('║  Checking every 5 minutes... Press Ctrl+C to stop              ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  while (true) {
    await checkAllVideos();
    console.log('\n⏳ Next check in 5 minutes...\n');
    await new Promise(r => setTimeout(r, 5 * 60 * 1000));
  }
}

function generateReport(): void {
  const log = loadPerformanceLog();
  
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📊 HEYGEN PERFORMANCE REPORT                                  ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`Generated: ${new Date().toISOString()}`);
  console.log(`Data from: ${log.last_updated}`);
  console.log('');
  
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('OVERALL STATISTICS');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`   Total Submitted:    ${log.stats.total_submitted}`);
  console.log(`   Completed:          ${log.stats.total_completed} (${(log.stats.total_completed / log.stats.total_submitted * 100).toFixed(1)}%)`);
  console.log(`   Pending:            ${log.stats.total_pending}`);
  console.log(`   Failed:             ${log.stats.total_failed}`);
  console.log('');
  
  if (log.stats.total_completed > 0) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('TIMING ANALYSIS');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`   Avg Queue Time:       ${log.stats.avg_queue_time_minutes} min (${(log.stats.avg_queue_time_minutes / 60).toFixed(1)} hrs)`);
    console.log(`   Avg Processing Time:  ${log.stats.avg_processing_time_minutes} min`);
    console.log(`   Avg Total Time:       ${log.stats.avg_total_time_minutes} min (${(log.stats.avg_total_time_minutes / 60).toFixed(1)} hrs)`);
    console.log('');
    
    // Find fastest and slowest
    const completedVideos = Object.values(log.videos).filter(v => v.status === 'completed' && v.total_time_minutes);
    if (completedVideos.length > 0) {
      const sorted = completedVideos.sort((a, b) => (a.total_time_minutes || 0) - (b.total_time_minutes || 0));
      const fastest = sorted[0];
      const slowest = sorted[sorted.length - 1];
      
      console.log('   Fastest: Day', fastest.day, '-', fastest.total_time_minutes, 'min');
      console.log('   Slowest: Day', slowest.day, '-', slowest.total_time_minutes, 'min');
    }
  }
  
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('COMPLETED VIDEOS');
  console.log('═══════════════════════════════════════════════════════════════');
  
  const completed = Object.values(log.videos)
    .filter(v => v.status === 'completed')
    .sort((a, b) => a.day - b.day);
  
  if (completed.length === 0) {
    console.log('   No completed videos yet');
  } else {
    for (const v of completed.slice(0, 20)) {
      console.log(`   Day ${v.day}: ${v.total_time_minutes} min - ${v.video_url?.substring(0, 50)}...`);
    }
    if (completed.length > 20) {
      console.log(`   ... and ${completed.length - 20} more`);
    }
  }
  
  console.log('');
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN PERFORMANCE MONITOR                                 ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  if (!HEYGEN_API_KEY) {
    console.error('\n❌ HEYGEN_API_KEY not found in environment');
    process.exit(1);
  }
  
  const args = process.argv.slice(2);
  
  if (args.includes('--report')) {
    generateReport();
  } else if (args.includes('--watch')) {
    await watchMode();
  } else {
    await checkAllVideos();
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
