/**
 * SPRINT 2.1 - Upload all downloaded HeyGen videos to Vercel Blob
 * 
 * Reads from: C:\Users\user\kelly-pipeline\heygen-downloads\day-{NNN}\{phase}-{age}.mp4
 * Uploads to: Vercel Blob at kelly-videos/day-{NNN}/{phase}-{age}.mp4
 * 
 * RESUME SUPPORT: Tracks uploaded files in upload-manifest.json, skips already-uploaded.
 * Stores new Blob URLs for database update in Sprint 2.2.
 */

import { config } from 'dotenv';
config();

import { put } from '@vercel/blob';
import * as fs from 'fs';
import * as path from 'path';

const DOWNLOAD_DIR = 'C:\\Users\\user\\kelly-pipeline\\heygen-downloads';
const MANIFEST_PATH = path.join(DOWNLOAD_DIR, 'blob-upload-manifest.json');
const CONCURRENT_UPLOADS = 3; // Conservative to avoid rate limits

interface UploadRecord {
  localPath: string;
  blobPath: string;
  blobUrl: string;
  size: number;
  day: number;
  phase: string;
  ageCategory: string;
  uploadedAt: string;
}

interface Manifest {
  uploads: UploadRecord[];
  lastUpdated: string;
  totalUploaded: number;
  totalFailed: number;
}

function loadManifest(): Manifest {
  if (fs.existsSync(MANIFEST_PATH)) {
    return JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
  }
  return { uploads: [], lastUpdated: '', totalUploaded: 0, totalFailed: 0 };
}

function saveManifest(manifest: Manifest) {
  manifest.lastUpdated = new Date().toISOString();
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
}

async function main() {
  console.log('=== SPRINT 2.1: UPLOAD TO VERCEL BLOB ===\n');

  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('FATAL: BLOB_READ_WRITE_TOKEN not set. Check .env file.');
    process.exit(1);
  }

  const manifest = loadManifest();
  const alreadyUploaded = new Set(manifest.uploads.map(u => u.localPath));
  
  console.log(`Previously uploaded: ${alreadyUploaded.size}`);

  // Find all MP4 files to upload
  const dayDirs = fs.readdirSync(DOWNLOAD_DIR)
    .filter(d => d.startsWith('day-'))
    .sort();

  const filesToUpload: { localPath: string; blobPath: string; day: number; phase: string; age: string }[] = [];

  for (const dayDir of dayDirs) {
    const dayPath = path.join(DOWNLOAD_DIR, dayDir);
    if (!fs.statSync(dayPath).isDirectory()) continue;
    
    const dayNum = parseInt(dayDir.replace('day-', ''), 10);
    const files = fs.readdirSync(dayPath).filter(f => f.endsWith('.mp4'));
    
    for (const file of files) {
      const localPath = path.join(dayPath, file);
      
      if (alreadyUploaded.has(localPath)) continue;
      
      // Verify file is not truncated
      const stats = fs.statSync(localPath);
      if (stats.size < 10000) {
        console.warn(`  Skipping ${dayDir}/${file} — too small (${stats.size} bytes)`);
        continue;
      }

      // Parse filename: phase-age.mp4
      const match = file.match(/^(\w+)-(\w+)\.mp4$/);
      if (!match) {
        console.warn(`  Skipping ${dayDir}/${file} — can't parse filename`);
        continue;
      }

      const [, phase, age] = match;
      const blobPath = `kelly-videos/${dayDir}/${file}`;

      filesToUpload.push({ localPath, blobPath, day: dayNum, phase, age });
    }
  }

  console.log(`Files to upload: ${filesToUpload.length}\n`);

  if (filesToUpload.length === 0) {
    console.log('✅ All files already uploaded!');
    return;
  }

  let succeeded = 0;
  let failed = 0;
  const failures: { file: string; error: string }[] = [];
  const startTime = Date.now();

  // Upload in batches
  for (let i = 0; i < filesToUpload.length; i += CONCURRENT_UPLOADS) {
    const batch = filesToUpload.slice(i, i + CONCURRENT_UPLOADS);
    
    const results = await Promise.allSettled(
      batch.map(async (file) => {
        const buffer = fs.readFileSync(file.localPath);
        
        const blob = await put(file.blobPath, buffer, {
          access: 'public',
          contentType: 'video/mp4',
          token: process.env.BLOB_READ_WRITE_TOKEN!,
        });

        return { ...file, blobUrl: blob.url, size: buffer.length };
      })
    );

    for (let j = 0; j < results.length; j++) {
      const result = results[j];
      const file = batch[j];

      if (result.status === 'fulfilled') {
        succeeded++;
        const { blobUrl, size, day, phase, age } = result.value;
        
        manifest.uploads.push({
          localPath: file.localPath,
          blobPath: file.blobPath,
          blobUrl,
          size,
          day,
          phase,
          ageCategory: age,
          uploadedAt: new Date().toISOString(),
        });
        
        const sizeMB = (size / (1024 * 1024)).toFixed(1);
        process.stdout.write(`\r✅ ${succeeded + failed}/${filesToUpload.length} | Day ${day} ${phase}-${age} (${sizeMB}MB) → Blob      `);
      } else {
        failed++;
        const error = (result.reason as Error).message;
        failures.push({ file: file.blobPath, error });
        process.stdout.write(`\r❌ ${succeeded + failed}/${filesToUpload.length} | ${file.blobPath}: ${error}      `);
      }
    }

    // Save manifest after each batch (resume support)
    manifest.totalUploaded = manifest.uploads.length;
    manifest.totalFailed = failed;
    saveManifest(manifest);
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);

  console.log(`\n\n════════════════════════════════════════`);
  console.log(`   UPLOAD COMPLETE`);
  console.log(`════════════════════════════════════════`);
  console.log(`Succeeded:         ${succeeded}`);
  console.log(`Failed:            ${failed}`);
  console.log(`Previously done:   ${alreadyUploaded.size}`);
  console.log(`Total in Blob:     ${manifest.uploads.length}`);
  console.log(`Time:              ${elapsed}s`);

  if (failures.length > 0) {
    console.log('\nFailed uploads:');
    for (const f of failures) {
      console.log(`  ${f.file}: ${f.error}`);
    }
  }

  // Show some sample URLs
  console.log('\nSample Blob URLs:');
  manifest.uploads.slice(0, 5).forEach(u => {
    console.log(`  Day ${u.day} ${u.phase}: ${u.blobUrl}`);
  });

  console.log(`\nManifest saved: ${MANIFEST_PATH}`);
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
