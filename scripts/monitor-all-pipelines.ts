#!/usr/bin/env npx tsx
/**
 * 📊 MASTER PIPELINE MONITOR
 * 
 * Monitors all active video generation pipelines:
 * - Sync Labs (from batch-lipsync-pipeline.ts)
 * - HeyGen (from heygen-use-all-credits.ts)
 * - Database video counts
 * 
 * Usage:
 *   npx tsx scripts/monitor-all-pipelines.ts
 *   npx tsx scripts/monitor-all-pipelines.ts --loop  (refresh every 30s)
 */

import 'dotenv/config';
import { config } from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';

config({ path: '.env.local' });
config({ path: '.env' });

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

interface PipelineStatus {
  name: string;
  status: 'running' | 'completed' | 'unknown';
  progress?: string;
  details?: string;
}

async function checkSyncLabsProgress(): Promise<PipelineStatus> {
  const progressFile = path.join(process.cwd(), 'logs', 'lipsync-progress.json');
  
  try {
    if (!fs.existsSync(progressFile)) {
      return { name: 'Sync Labs', status: 'unknown', details: 'No progress file found' };
    }
    
    const data = JSON.parse(fs.readFileSync(progressFile, 'utf-8'));
    const { completed, total, startedAt, lastUpdated } = data;
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    const lastUpdate = new Date(lastUpdated);
    const now = new Date();
    const minutesAgo = Math.round((now.getTime() - lastUpdate.getTime()) / 60000);
    
    const isRunning = minutesAgo < 10; // Consider running if updated in last 10 min
    
    return {
      name: 'Sync Labs',
      status: isRunning ? 'running' : 'completed',
      progress: `${completed}/${total} (${percentage}%)`,
      details: `Last update: ${minutesAgo}m ago`
    };
  } catch (err) {
    return { name: 'Sync Labs', status: 'unknown', details: 'Error reading progress' };
  }
}

async function checkHeyGenCredits(): Promise<PipelineStatus> {
  if (!HEYGEN_API_KEY) {
    return { name: 'HeyGen', status: 'unknown', details: 'No API key' };
  }
  
  try {
    const response = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    
    const data = await response.json();
    const credits = data?.data?.remaining_quota || 0;
    
    // Check for recent jobs file
    const today = new Date().toISOString().split('T')[0];
    const jobsFile = `heygen-jobs-${today}.json`;
    let jobsSubmitted = 0;
    
    if (fs.existsSync(jobsFile)) {
      const jobs = JSON.parse(fs.readFileSync(jobsFile, 'utf-8'));
      jobsSubmitted = jobs.length;
    }
    
    return {
      name: 'HeyGen',
      status: credits > 0 ? 'running' : 'completed',
      progress: `${credits} credits remaining`,
      details: jobsSubmitted > 0 ? `${jobsSubmitted} jobs submitted today` : 'No jobs today'
    };
  } catch (err: any) {
    return { name: 'HeyGen', status: 'unknown', details: err.message };
  }
}

async function checkDatabaseVideos(): Promise<PipelineStatus> {
  // Read from Neon directly if we have the connection string
  const dbUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;
  
  if (!dbUrl) {
    return { name: 'Database', status: 'unknown', details: 'No DB connection' };
  }
  
  try {
    // Use pg to query
    const pg = await import('pg');
    const pool = new pg.default.Pool({
      connectionString: dbUrl,
      ssl: { rejectUnauthorized: false }
    });
    
    const result = await pool.query(`
      SELECT 
        COUNT(*) as total,
        COUNT(video_url) as with_video,
        COUNT(audio_url) as with_audio
      FROM kelly_lesson_assets
    `);
    
    await pool.end();
    
    const { total, with_video, with_audio } = result.rows[0];
    
    return {
      name: 'Database',
      status: 'running',
      progress: `${with_video} videos, ${with_audio} audio`,
      details: `Total records: ${total}`
    };
  } catch (err: any) {
    return { name: 'Database', status: 'unknown', details: err.message };
  }
}

async function checkAntigravityStatus(): Promise<PipelineStatus> {
  // Check for Antigravity output files
  const antigravityPath = 'C:\\Users\\user\\ANTIGRAVITY\\outputs';
  
  try {
    if (!fs.existsSync(antigravityPath)) {
      return { name: 'Antigravity', status: 'unknown', details: 'Output folder not found' };
    }
    
    const files = fs.readdirSync(antigravityPath);
    const jsonFiles = files.filter(f => f.endsWith('.json'));
    
    // Get most recent file
    let mostRecent = '';
    let mostRecentTime = 0;
    
    for (const file of jsonFiles) {
      const stat = fs.statSync(path.join(antigravityPath, file));
      if (stat.mtimeMs > mostRecentTime) {
        mostRecentTime = stat.mtimeMs;
        mostRecent = file;
      }
    }
    
    const minutesAgo = Math.round((Date.now() - mostRecentTime) / 60000);
    
    return {
      name: 'Antigravity',
      status: minutesAgo < 30 ? 'running' : 'completed',
      progress: `${jsonFiles.length} output files`,
      details: mostRecent ? `Latest: ${mostRecent} (${minutesAgo}m ago)` : 'No recent output'
    };
  } catch (err: any) {
    return { name: 'Antigravity', status: 'unknown', details: err.message };
  }
}

function printStatus(pipelines: PipelineStatus[]) {
  console.clear();
  console.log('═'.repeat(60));
  console.log('  📊 KELLY VIDEO PIPELINE MONITOR');
  console.log('  ' + new Date().toLocaleString());
  console.log('═'.repeat(60));
  console.log();
  
  for (const p of pipelines) {
    const statusIcon = {
      running: '🟢',
      completed: '✅',
      unknown: '⚪'
    }[p.status];
    
    console.log(`${statusIcon} ${p.name.padEnd(15)} ${p.status.toUpperCase()}`);
    if (p.progress) console.log(`   Progress: ${p.progress}`);
    if (p.details) console.log(`   Details:  ${p.details}`);
    console.log();
  }
  
  console.log('─'.repeat(60));
  console.log('Commands:');
  console.log('  npx tsx scripts/heygen-use-all-credits.ts --days=1-7');
  console.log('  npx tsx scripts/batch-lipsync-pipeline.ts --status');
  console.log('  npx tsx scripts/download-heygen-videos.ts --audit');
  console.log('─'.repeat(60));
}

async function main() {
  const args = process.argv.slice(2);
  const loop = args.includes('--loop');
  
  const runOnce = async () => {
    const pipelines = await Promise.all([
      checkSyncLabsProgress(),
      checkHeyGenCredits(),
      checkDatabaseVideos(),
      checkAntigravityStatus()
    ]);
    
    printStatus(pipelines);
  };
  
  if (loop) {
    console.log('Monitoring... (Ctrl+C to stop)\n');
    while (true) {
      await runOnce();
      await new Promise(r => setTimeout(r, 30000)); // Refresh every 30s
    }
  } else {
    await runOnce();
  }
}

main().catch(console.error);
