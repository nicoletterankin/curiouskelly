#!/usr/bin/env npx tsx
/**
 * BACKUP HEYGEN VIDEOS
 * 
 * Downloads all HeyGen videos before their URLs expire.
 * Videos are stored in video-backups/ for permanent storage.
 * 
 * Usage:
 *   npx tsx scripts/backup-heygen-videos.ts
 *   npx tsx scripts/backup-heygen-videos.ts --day 351
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

const BACKUP_DIR = path.join(process.cwd(), 'video-backups');

interface VideoManifestEntry {
  video_id: string;
  status: string;
  video_url?: string;
  phases?: string[];
  estimated_duration?: number;
}

interface DayManifest {
  day: number;
  videos: Record<string, VideoManifestEntry>;
}

async function downloadFile(url: string, destPath: string): Promise<boolean> {
  return new Promise((resolve) => {
    const file = fs.createWriteStream(destPath);
    
    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Follow redirect
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          https.get(redirectUrl, (res) => {
            res.pipe(file);
            file.on('finish', () => {
              file.close();
              resolve(true);
            });
          }).on('error', () => resolve(false));
        } else {
          resolve(false);
        }
        return;
      }
      
      if (response.statusCode !== 200) {
        console.error(`   ❌ HTTP ${response.statusCode}`);
        resolve(false);
        return;
      }
      
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve(true);
      });
    }).on('error', (err) => {
      console.error(`   ❌ Download error:`, err.message);
      fs.unlink(destPath, () => {});
      resolve(false);
    });
  });
}

async function backupDay(dayNumber: number): Promise<{ downloaded: number; failed: number }> {
  const manifestPath = path.join(process.cwd(), 'generated-videos', `day-${dayNumber}-manifest.json`);
  
  if (!fs.existsSync(manifestPath)) {
    console.log(`⏭️  Day ${dayNumber}: No manifest found`);
    return { downloaded: 0, failed: 0 };
  }
  
  const manifest: DayManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  const dayDir = path.join(BACKUP_DIR, `day-${String(dayNumber).padStart(3, '0')}`);
  
  if (!fs.existsSync(dayDir)) {
    fs.mkdirSync(dayDir, { recursive: true });
  }
  
  console.log(`\n📦 Day ${dayNumber}: Processing ${Object.keys(manifest.videos).length} video(s)`);
  
  let downloaded = 0;
  let failed = 0;
  
  for (const [archetype, video] of Object.entries(manifest.videos)) {
    if (video.status !== 'completed' || !video.video_url) {
      console.log(`   ⏭️  ${archetype}: Not ready (${video.status})`);
      continue;
    }
    
    const filename = `${archetype}.mp4`;
    const destPath = path.join(dayDir, filename);
    
    // Check if already downloaded
    if (fs.existsSync(destPath)) {
      const stats = fs.statSync(destPath);
      if (stats.size > 1000000) { // > 1MB = probably valid
        console.log(`   ✅ ${archetype}: Already backed up (${(stats.size / 1024 / 1024).toFixed(1)}MB)`);
        downloaded++;
        continue;
      }
    }
    
    console.log(`   ⬇️  ${archetype}: Downloading...`);
    
    const success = await downloadFile(video.video_url, destPath);
    
    if (success) {
      const stats = fs.statSync(destPath);
      console.log(`   ✅ ${archetype}: Downloaded (${(stats.size / 1024 / 1024).toFixed(1)}MB)`);
      downloaded++;
    } else {
      console.log(`   ❌ ${archetype}: Failed`);
      failed++;
    }
  }
  
  return { downloaded, failed };
}

async function backupSummaryVideo(dayNumber: number): Promise<boolean> {
  const manifestPath = path.join(process.cwd(), 'content', 'email-summary-video', `day-${dayNumber}-summary-manifest.json`);
  
  if (!fs.existsSync(manifestPath)) {
    return false;
  }
  
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  if (!manifest.video_url) {
    return false;
  }
  
  const dayDir = path.join(BACKUP_DIR, `day-${String(dayNumber).padStart(3, '0')}`);
  if (!fs.existsSync(dayDir)) {
    fs.mkdirSync(dayDir, { recursive: true });
  }
  
  const destPath = path.join(dayDir, 'summary.mp4');
  
  // Check if already downloaded
  if (fs.existsSync(destPath)) {
    const stats = fs.statSync(destPath);
    if (stats.size > 1000000) {
      console.log(`   ✅ summary: Already backed up (${(stats.size / 1024 / 1024).toFixed(1)}MB)`);
      return true;
    }
  }
  
  console.log(`   ⬇️  summary: Downloading...`);
  
  // Add signed URL params if not present
  let url = manifest.video_url;
  if (!url.includes('Expires=')) {
    // Try to get fresh URL from HeyGen
    console.log(`   ⚠️  summary: URL may be expired, trying anyway...`);
  }
  
  const success = await downloadFile(url, destPath);
  
  if (success) {
    const stats = fs.statSync(destPath);
    console.log(`   ✅ summary: Downloaded (${(stats.size / 1024 / 1024).toFixed(1)}MB)`);
    
    // Also download thumbnail
    if (manifest.thumbnail_url) {
      const thumbPath = path.join(dayDir, 'summary-thumb.jpg');
      await downloadFile(manifest.thumbnail_url, thumbPath);
    }
    
    return true;
  }
  
  return false;
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  💾 HEYGEN VIDEO BACKUP                                        ║');
  console.log('║  Download and preserve all rendered videos                     ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  // Create backup directory
  if (!fs.existsSync(BACKUP_DIR)) {
    fs.mkdirSync(BACKUP_DIR, { recursive: true });
  }
  
  // Check for specific day argument
  const args = process.argv.slice(2);
  let specificDay: number | null = null;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      specificDay = parseInt(args[i + 1]);
    }
  }
  
  if (specificDay) {
    console.log(`\n🎯 Backing up Day ${specificDay} only`);
    await backupSummaryVideo(specificDay);
    await backupDay(specificDay);
    return;
  }
  
  // Find all manifest files
  const manifestDir = path.join(process.cwd(), 'generated-videos');
  const manifestFiles = fs.readdirSync(manifestDir)
    .filter(f => f.match(/day-\d+-manifest\.json/));
  
  console.log(`\n📋 Found ${manifestFiles.length} day manifest(s)`);
  
  let totalDownloaded = 0;
  let totalFailed = 0;
  
  for (const file of manifestFiles) {
    const dayMatch = file.match(/day-(\d+)-manifest/);
    if (!dayMatch) continue;
    
    const dayNumber = parseInt(dayMatch[1]);
    
    // Backup summary video first
    await backupSummaryVideo(dayNumber);
    
    // Then backup archetype videos
    const result = await backupDay(dayNumber);
    totalDownloaded += result.downloaded;
    totalFailed += result.failed;
  }
  
  console.log('');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`✅ BACKUP COMPLETE`);
  console.log(`   Downloaded: ${totalDownloaded}`);
  console.log(`   Failed: ${totalFailed}`);
  console.log(`   Location: ${BACKUP_DIR}`);
  console.log('════════════════════════════════════════════════════════════════');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
