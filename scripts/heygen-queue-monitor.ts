#!/usr/bin/env npx tsx
/**
 * 🔍 HEYGEN QUEUE MONITOR
 * Checks status of all pending HeyGen videos and updates manifests
 */
import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

interface VideoStatus {
  video_id: string;
  status: string;
  video_url?: string;
  error?: string;
}

async function checkVideoStatus(videoId: string): Promise<VideoStatus> {
  const response = await fetch(
    `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
    { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
  );
  
  if (!response.ok) {
    return { video_id: videoId, status: 'error', error: `HTTP ${response.status}` };
  }
  
  const data = await response.json();
  return {
    video_id: videoId,
    status: data.data?.status || 'unknown',
    video_url: data.data?.video_url
  };
}

async function monitorDay351() {
  console.log('\n📹 DAY 351 MANIFEST');
  console.log('═'.repeat(50));
  
  const manifestPath = 'generated-videos/day-351-manifest.json';
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  let updated = false;
  let completed = 0;
  let waiting = 0;
  let pending = 0;
  
  for (const [archetype, video] of Object.entries(manifest.videos) as [string, any][]) {
    if (!video.video_id) continue;
    
    const status = await checkVideoStatus(video.video_id);
    const icon = status.status === 'completed' ? '✅' : 
                 status.status === 'waiting' ? '⏳' : 
                 status.status === 'pending' ? '🔄' : '❌';
    
    console.log(`  ${icon} ${archetype.padEnd(12)} ${status.status}`);
    
    if (status.status === 'completed') completed++;
    else if (status.status === 'waiting') waiting++;
    else if (status.status === 'pending') pending++;
    
    // Update manifest if status changed
    if (video.status !== status.status || (status.video_url && !video.video_url)) {
      manifest.videos[archetype].status = status.status;
      if (status.video_url) {
        manifest.videos[archetype].video_url = status.video_url;
      }
      updated = true;
    }
  }
  
  if (updated) {
    manifest.updated = new Date().toISOString();
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    console.log('\n  📝 Manifest updated');
  }
  
  console.log(`\n  Summary: ${completed} completed, ${waiting} waiting, ${pending} pending`);
  return { completed, waiting, pending };
}

async function monitorBatchQueue() {
  console.log('\n📹 BATCH QUEUE (Days 1-365)');
  console.log('═'.repeat(50));
  
  const perfPath = 'logs/heygen/performance.json';
  if (!fs.existsSync(perfPath)) {
    console.log('  No batch queue file found');
    return;
  }
  
  const perf = JSON.parse(fs.readFileSync(perfPath, 'utf-8'));
  const videos = Object.entries(perf.videos) as [string, any][];
  
  // Sample check (first 10 and last 10)
  const sample = [...videos.slice(0, 10), ...videos.slice(-10)];
  
  let completed = 0;
  let waiting = 0;
  let failed = 0;
  let updated = false;
  
  console.log('  Checking sample of 20 videos...');
  
  for (const [key, video] of sample) {
    const status = await checkVideoStatus(video.video_id);
    
    if (status.status === 'completed') {
      completed++;
      if (video.status !== 'completed') {
        perf.videos[key].status = 'completed';
        perf.videos[key].video_url = status.video_url;
        updated = true;
      }
    } else if (status.status === 'waiting') {
      waiting++;
    } else {
      failed++;
    }
  }
  
  if (updated) {
    perf.last_updated = new Date().toISOString();
    fs.writeFileSync(perfPath, JSON.stringify(perf, null, 2));
    console.log('  📝 Performance log updated');
  }
  
  console.log(`\n  Sample: ${completed} completed, ${waiting} waiting, ${failed} failed`);
  console.log(`  Total in queue: ${videos.length} videos`);
  
  // Estimate based on sample
  const completionRate = completed / sample.length;
  const estimatedComplete = Math.round(videos.length * completionRate);
  console.log(`  Estimated total completed: ~${estimatedComplete}`);
}

async function main() {
  console.log('🔍 HEYGEN QUEUE MONITOR');
  console.log('═'.repeat(50));
  console.log(`Time: ${new Date().toISOString()}`);
  console.log(`API Key: ${HEYGEN_API_KEY?.substring(0, 20)}...`);
  
  // Check quota
  const quotaRes = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  if (quotaRes.ok) {
    const quota = await quotaRes.json();
    console.log(`Credits: ${quota.data?.remaining_quota || 'unknown'}`);
  }
  
  // Monitor Day 351
  const day351 = await monitorDay351();
  
  // Monitor batch queue
  await monitorBatchQueue();
  
  console.log('\n' + '═'.repeat(50));
  console.log('✅ Monitor complete');
  
  // Return exit code based on Day 351 status
  if (day351.completed === 12) {
    console.log('🎉 Day 351 fully complete!');
    process.exit(0);
  } else if (day351.waiting + day351.pending > 0) {
    console.log(`⏳ ${day351.waiting + day351.pending} videos still processing`);
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
