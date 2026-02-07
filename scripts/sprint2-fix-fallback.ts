/**
 * SPRINT 2 - Fix dead fallback base videos
 * 
 * The API route constructs fallback URLs like:
 *   video/kelly/base/adult-storyteller.mp4
 *   video/kelly/base/kid-storyteller.mp4
 *   video/kelly/base/senior-storyteller.mp4
 * 
 * These don't exist in Blob. Upload actual Kelly base videos to these paths.
 * Uses existing base expression videos from kelly-pipeline/videos/base/
 */

import { config } from 'dotenv';
config();

import { put } from '@vercel/blob';
import * as fs from 'fs';
import * as path from 'path';

const BASE_DIR = 'C:\\Users\\user\\kelly-pipeline\\videos\\base';
const ARCHETYPES = ['storyteller', 'explorer', 'scientist', 'architect', 'strategist', 
  'diplomat', 'mystic', 'rebel', 'macgyver', 'empath', 'provider', 'survivor'];
const AGE_GROUPS = ['adult', 'kid', 'senior'];

async function main() {
  console.log('=== FIX FALLBACK BASE VIDEOS ===\n');

  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('FATAL: BLOB_READ_WRITE_TOKEN not set.');
    process.exit(1);
  }

  // Find available base videos by expression
  const expressionDirs = ['talking', 'excited', 'curious', 'thinking'];
  const sourceVideos: Record<string, string> = {};

  for (const age of ['adult', 'elder', 'kid', 'teen']) {
    const agePath = path.join(BASE_DIR, age);
    if (!fs.existsSync(agePath)) continue;
    
    for (const expr of expressionDirs) {
      const exprPath = path.join(agePath, expr);
      if (!fs.existsSync(exprPath)) continue;
      
      const files = fs.readdirSync(exprPath).filter(f => f.endsWith('.mp4'));
      if (files.length > 0) {
        // Pick the largest file (likely best quality)
        const sorted = files
          .map(f => ({ name: f, size: fs.statSync(path.join(exprPath, f)).size }))
          .sort((a, b) => b.size - a.size);
        
        sourceVideos[`${age}-${expr}`] = path.join(exprPath, sorted[0].name);
        console.log(`  Found: ${age}/${expr} → ${sorted[0].name} (${(sorted[0].size / 1024 / 1024).toFixed(1)}MB)`);
      }
    }
  }

  // Also check Downloads for the named base files
  const downloadsBase = 'C:\\Users\\user\\Downloads\\Kelly character bible video base 1.mp4';
  if (fs.existsSync(downloadsBase)) {
    sourceVideos['downloads-base'] = downloadsBase;
    console.log(`  Found: Downloads base video (${(fs.statSync(downloadsBase).size / 1024 / 1024).toFixed(1)}MB)`);
  }

  console.log(`\nTotal source videos found: ${Object.keys(sourceVideos).length}\n`);

  // Upload strategy: For each age group × archetype, upload a base video
  // Use the "talking" expression as the default for all archetypes
  const uploads: { blobPath: string; sourcePath: string }[] = [];

  for (const age of AGE_GROUPS) {
    // Find best source for this age
    let source: string | null = null;
    
    // Try exact age match with 'talking' expression
    if (sourceVideos[`${age}-talking`]) source = sourceVideos[`${age}-talking`];
    // Fallback to 'excited'
    else if (sourceVideos[`${age}-excited`]) source = sourceVideos[`${age}-excited`];
    // Fallback to adult
    else if (sourceVideos['adult-talking']) source = sourceVideos['adult-talking'];
    // Last resort
    else source = Object.values(sourceVideos)[0];

    if (!source) {
      console.warn(`No source video for age group: ${age}`);
      continue;
    }

    for (const archetype of ARCHETYPES) {
      uploads.push({
        blobPath: `video/kelly/base/${age}-${archetype}.mp4`,
        sourcePath: source,
      });
    }
  }

  console.log(`Uploading ${uploads.length} base video variants...\n`);

  let succeeded = 0;
  let failed = 0;

  for (const upload of uploads) {
    try {
      const buffer = fs.readFileSync(upload.sourcePath);
      const blob = await put(upload.blobPath, buffer, {
        access: 'public',
        contentType: 'video/mp4',
        token: process.env.BLOB_READ_WRITE_TOKEN!,
      });
      succeeded++;
      if (succeeded % 10 === 0 || succeeded <= 3) {
        console.log(`  ✅ ${succeeded}/${uploads.length} | ${upload.blobPath} → ${blob.url.substring(0, 60)}...`);
      }
    } catch (err) {
      failed++;
      console.error(`  ❌ ${upload.blobPath}: ${(err as Error).message}`);
    }
  }

  console.log(`\nDone: ${succeeded} uploaded, ${failed} failed`);

  // Verify one of the fallback URLs
  console.log('\nVerifying fallback URLs...');
  const testUrl = 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/video/kelly/base/adult-storyteller.mp4';
  try {
    const resp = await fetch(testUrl, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
    console.log(`  adult-storyteller: ${resp.status} (${resp.headers.get('content-type')}, ${(parseInt(resp.headers.get('content-length') || '0') / 1024 / 1024).toFixed(1)}MB)`);
  } catch (err) {
    console.log(`  adult-storyteller: ERROR - ${(err as Error).message}`);
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
