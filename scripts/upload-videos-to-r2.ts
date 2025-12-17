#!/usr/bin/env npx tsx
/**
 * UPLOAD VIDEOS TO CLOUDFLARE R2
 * 
 * Uploads backed-up videos to permanent R2 storage
 * 
 * Usage:
 *   npx tsx scripts/upload-videos-to-r2.ts
 *   npx tsx scripts/upload-videos-to-r2.ts --day 351
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { S3Client, PutObjectCommand, HeadObjectCommand } from '@aws-sdk/client-s3';

const BACKUP_DIR = path.join(process.cwd(), 'video-backups');

// R2 Configuration from .env
const R2_ENDPOINT = process.env.CLOUDFLARE_R2_ENDPOINT;
const R2_ACCESS_KEY = process.env.CLOUDFLARE_R2_ACCESS_KEY;
const R2_SECRET_KEY = process.env.CLOUDFLARE_R2_SECRET_KEY;
const R2_BUCKET = process.env.CLOUDFLARE_R2_BUCKET || 'curious-kelly-backups';

// Public URL base (we'll construct this from the bucket)
const R2_PUBLIC_BASE = `https://pub-${R2_BUCKET}.r2.dev`;

function getR2Client(): S3Client {
  if (!R2_ENDPOINT || !R2_ACCESS_KEY || !R2_SECRET_KEY) {
    throw new Error('Missing R2 credentials in .env: CLOUDFLARE_R2_ENDPOINT, CLOUDFLARE_R2_ACCESS_KEY, CLOUDFLARE_R2_SECRET_KEY');
  }
  
  return new S3Client({
    region: 'auto',
    endpoint: R2_ENDPOINT,
    credentials: {
      accessKeyId: R2_ACCESS_KEY,
      secretAccessKey: R2_SECRET_KEY,
    },
  });
}

async function uploadFile(client: S3Client, localPath: string, r2Key: string): Promise<string | null> {
  try {
    const fileBuffer = fs.readFileSync(localPath);
    const contentType = localPath.endsWith('.mp4') ? 'video/mp4' : 
                       localPath.endsWith('.jpg') || localPath.endsWith('.jpeg') ? 'image/jpeg' :
                       'application/octet-stream';
    
    await client.send(new PutObjectCommand({
      Bucket: R2_BUCKET,
      Key: r2Key,
      Body: fileBuffer,
      ContentType: contentType,
    }));
    
    // Return the R2 URL (will need public access enabled on bucket)
    return `${R2_ENDPOINT?.replace('https://', 'https://pub-').replace('.r2.cloudflarestorage.com', '.r2.dev')}/${r2Key}`;
  } catch (error: any) {
    console.error(`   ❌ Upload failed: ${error.message}`);
    return null;
  }
}

async function checkExists(client: S3Client, r2Key: string): Promise<boolean> {
  try {
    await client.send(new HeadObjectCommand({
      Bucket: R2_BUCKET,
      Key: r2Key,
    }));
    return true;
  } catch {
    return false;
  }
}

async function uploadDay(client: S3Client, dayNumber: number): Promise<{ uploaded: number; skipped: number; failed: number }> {
  const dayDir = path.join(BACKUP_DIR, `day-${String(dayNumber).padStart(3, '0')}`);
  
  if (!fs.existsSync(dayDir)) {
    console.log(`⏭️  Day ${dayNumber}: No backup found`);
    return { uploaded: 0, skipped: 0, failed: 0 };
  }
  
  const files = fs.readdirSync(dayDir);
  console.log(`\n📤 Day ${dayNumber}: Uploading ${files.length} file(s)`);
  
  let uploaded = 0;
  let skipped = 0;
  let failed = 0;
  
  for (const file of files) {
    const localPath = path.join(dayDir, file);
    const stats = fs.statSync(localPath);
    const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
    
    // Determine R2 key based on file type
    let r2Key: string;
    if (file === 'summary.mp4') {
      r2Key = `videos/summary/day-${dayNumber}.mp4`;
    } else if (file === 'summary-thumb.jpg') {
      r2Key = `videos/summary/day-${dayNumber}-thumb.jpg`;
    } else if (file.endsWith('.mp4')) {
      const archetype = file.replace('.mp4', '');
      r2Key = `videos/full/day-${dayNumber}-${archetype}.mp4`;
    } else {
      continue; // Skip unknown files
    }
    
    // Check if already exists
    const exists = await checkExists(client, r2Key);
    if (exists) {
      console.log(`   ⏭️  ${file}: Already in R2`);
      skipped++;
      continue;
    }
    
    console.log(`   ⬆️  ${file} (${sizeMB}MB) → ${r2Key}`);
    
    const url = await uploadFile(client, localPath, r2Key);
    if (url) {
      console.log(`   ✅ Uploaded`);
      uploaded++;
    } else {
      failed++;
    }
  }
  
  return { uploaded, skipped, failed };
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  ☁️  UPLOAD VIDEOS TO CLOUDFLARE R2                             ║');
  console.log('║  Permanent, fast, free-egress video hosting                    ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  console.log(`\n📦 Bucket: ${R2_BUCKET}`);
  console.log(`🔗 Endpoint: ${R2_ENDPOINT}`);
  
  const client = getR2Client();
  
  // Check for specific day argument
  const args = process.argv.slice(2);
  let specificDay: number | null = null;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      specificDay = parseInt(args[i + 1]);
    }
  }
  
  if (specificDay) {
    console.log(`\n🎯 Uploading Day ${specificDay} only`);
    const result = await uploadDay(client, specificDay);
    console.log(`\n✅ Done: ${result.uploaded} uploaded, ${result.skipped} skipped, ${result.failed} failed`);
    return;
  }
  
  // Upload all days
  if (!fs.existsSync(BACKUP_DIR)) {
    console.log('❌ No video-backups directory found. Run backup-heygen-videos.ts first.');
    return;
  }
  
  const dayDirs = fs.readdirSync(BACKUP_DIR)
    .filter(d => d.match(/^day-\d+$/))
    .sort();
  
  console.log(`\n📋 Found ${dayDirs.length} day(s) to upload`);
  
  let totalUploaded = 0;
  let totalSkipped = 0;
  let totalFailed = 0;
  
  for (const dir of dayDirs) {
    const dayMatch = dir.match(/day-(\d+)/);
    if (!dayMatch) continue;
    
    const dayNumber = parseInt(dayMatch[1]);
    const result = await uploadDay(client, dayNumber);
    totalUploaded += result.uploaded;
    totalSkipped += result.skipped;
    totalFailed += result.failed;
  }
  
  console.log('');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`✅ UPLOAD COMPLETE`);
  console.log(`   Uploaded: ${totalUploaded}`);
  console.log(`   Skipped: ${totalSkipped}`);
  console.log(`   Failed: ${totalFailed}`);
  console.log('');
  console.log(`📍 Videos available at:`);
  console.log(`   ${R2_BUCKET}.r2.dev/videos/summary/day-XXX.mp4`);
  console.log('════════════════════════════════════════════════════════════════');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
