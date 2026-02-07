/**
 * SPRINT 2 - CRITICAL: Download all 513 HeyGen videos before expiry
 * 
 * Downloads all video URLs from heygen_videos table to local disk.
 * Files saved as: C:\Users\user\kelly-pipeline\heygen-downloads\day-{NNN}\{phase}-{age}.mp4
 * 
 * RESUME SUPPORT: Skips already-downloaded files.
 * PARALLEL: Downloads 5 at a time to balance speed vs bandwidth.
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';
import * as fs from 'fs';
import * as path from 'path';

const sql = neon(process.env.DATABASE_URL!);

const DOWNLOAD_DIR = 'C:\\Users\\user\\kelly-pipeline\\heygen-downloads';
const CONCURRENT_DOWNLOADS = 5;

interface VideoRecord {
  day_of_year: number;
  phase: string;
  video_url: string;
  age_category: string;
  archetype: string;
  status: string;
}

async function downloadFile(url: string, destPath: string): Promise<{ success: boolean; size: number; error?: string }> {
  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(120000) }); // 2 min timeout
    
    if (!resp.ok) {
      return { success: false, size: 0, error: `HTTP ${resp.status}` };
    }

    const buffer = Buffer.from(await resp.arrayBuffer());
    
    // Ensure directory exists
    const dir = path.dirname(destPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    fs.writeFileSync(destPath, buffer);
    return { success: true, size: buffer.length };
  } catch (err) {
    return { success: false, size: 0, error: (err as Error).message };
  }
}

async function main() {
  console.log('=== SPRINT 2: DOWNLOAD ALL HEYGEN VIDEOS ===\n');
  console.log(`Download directory: ${DOWNLOAD_DIR}`);
  console.log(`Concurrent downloads: ${CONCURRENT_DOWNLOADS}\n`);

  // Ensure base directory exists
  if (!fs.existsSync(DOWNLOAD_DIR)) {
    fs.mkdirSync(DOWNLOAD_DIR, { recursive: true });
  }

  // Get all videos
  const videos = await sql`
    SELECT day_of_year, phase, video_url, age_category, archetype, status
    FROM heygen_videos 
    WHERE video_url IS NOT NULL AND status = 'completed'
    ORDER BY day_of_year, phase
  ` as unknown as VideoRecord[];

  console.log(`Total videos to download: ${videos.length}\n`);

  // Build download list with resume support
  const downloadList: { record: VideoRecord; destPath: string }[] = [];
  let alreadyDownloaded = 0;

  for (const video of videos) {
    const dayDir = `day-${String(video.day_of_year).padStart(3, '0')}`;
    const fileName = `${video.phase}-${video.age_category || 'adult'}.mp4`;
    const destPath = path.join(DOWNLOAD_DIR, dayDir, fileName);

    if (fs.existsSync(destPath)) {
      const stats = fs.statSync(destPath);
      if (stats.size > 10000) { // > 10KB = not truncated
        alreadyDownloaded++;
        continue;
      }
    }

    downloadList.push({ record: video, destPath });
  }

  console.log(`Already downloaded: ${alreadyDownloaded}`);
  console.log(`Remaining to download: ${downloadList.length}\n`);

  if (downloadList.length === 0) {
    console.log('✅ All videos already downloaded!');
    return;
  }

  // Download in batches
  let succeeded = 0;
  let failed = 0;
  const failures: { day: number; phase: string; error: string }[] = [];
  const startTime = Date.now();

  for (let i = 0; i < downloadList.length; i += CONCURRENT_DOWNLOADS) {
    const batch = downloadList.slice(i, i + CONCURRENT_DOWNLOADS);
    
    const results = await Promise.all(
      batch.map(async ({ record, destPath }) => {
        const result = await downloadFile(record.video_url, destPath);
        return { record, destPath, result };
      })
    );

    for (const { record, result } of results) {
      if (result.success) {
        succeeded++;
        const sizeMB = (result.size / (1024 * 1024)).toFixed(1);
        process.stdout.write(`\r✅ ${succeeded + failed}/${downloadList.length} | Day ${record.day_of_year} ${record.phase} (${sizeMB}MB)      `);
      } else {
        failed++;
        failures.push({ day: record.day_of_year, phase: record.phase, error: result.error || 'unknown' });
        process.stdout.write(`\r❌ ${succeeded + failed}/${downloadList.length} | Day ${record.day_of_year} ${record.phase}: ${result.error}      `);
      }
    }
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  
  console.log(`\n\n════════════════════════════════════════`);
  console.log(`   DOWNLOAD COMPLETE`);
  console.log(`════════════════════════════════════════`);
  console.log(`Succeeded:         ${succeeded}`);
  console.log(`Failed:            ${failed}`);
  console.log(`Already had:       ${alreadyDownloaded}`);
  console.log(`Total local:       ${succeeded + alreadyDownloaded}`);
  console.log(`Time:              ${elapsed}s`);
  
  if (failures.length > 0) {
    console.log(`\nFailed downloads:`);
    for (const f of failures) {
      console.log(`  Day ${f.day} ${f.phase}: ${f.error}`);
    }
  }

  // Save download manifest
  const manifest = {
    timestamp: new Date().toISOString(),
    total: videos.length,
    downloaded: succeeded + alreadyDownloaded,
    newlyDownloaded: succeeded,
    failed: failed,
    failures,
    directory: DOWNLOAD_DIR,
  };
  
  fs.writeFileSync(
    path.join(DOWNLOAD_DIR, 'download-manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
  console.log(`\nManifest saved: ${path.join(DOWNLOAD_DIR, 'download-manifest.json')}`);
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
