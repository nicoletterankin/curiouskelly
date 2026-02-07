/**
 * SPRINT 2.2 - Update heygen_videos table with permanent Vercel Blob URLs
 * 
 * Reads the blob upload manifest and updates each matching row in heygen_videos
 * to point to the permanent Blob URL instead of the expiring HeyGen CDN URL.
 * 
 * Strategy:
 * - For each uploaded video (day, phase, age_category)
 * - Find matching rows in heygen_videos  
 * - Update video_url to the new Blob URL
 * - Keep the old HeyGen URL in a backup column (or log it)
 */

import { config } from 'dotenv';
config();

import { neon } from '@neondatabase/serverless';
import * as fs from 'fs';

const sql = neon(process.env.DATABASE_URL!);

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
}

async function main() {
  console.log('=== SPRINT 2.2: UPDATE DATABASE WITH BLOB URLs ===\n');

  const manifestPath = 'C:\\Users\\user\\kelly-pipeline\\heygen-downloads\\blob-upload-manifest.json';
  const manifest: Manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  console.log(`Upload manifest entries: ${manifest.uploads.length}\n`);

  // First, check if heygen_videos has a column for the old URL backup
  console.log('Checking table structure...');
  const columns = await sql`
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'heygen_videos'
    ORDER BY ordinal_position
  `;
  const columnNames = columns.map((c: any) => c.column_name);
  console.log(`Columns: ${columnNames.join(', ')}`);

  // Add a backup column if it doesn't exist
  if (!columnNames.includes('original_heygen_url')) {
    console.log('\nAdding original_heygen_url backup column...');
    try {
      await sql`ALTER TABLE heygen_videos ADD COLUMN IF NOT EXISTS original_heygen_url TEXT`;
      console.log('  Column added.');
    } catch (err) {
      console.warn('  Could not add column (may already exist):', (err as Error).message);
    }
  }

  // Group uploads by day+phase+age for efficient updates
  // For each upload, update all matching rows
  let updated = 0;
  let skipped = 0;
  let errors = 0;

  // Backup all current HeyGen URLs first
  console.log('\nBacking up current HeyGen URLs...');
  try {
    await sql`
      UPDATE heygen_videos 
      SET original_heygen_url = video_url 
      WHERE original_heygen_url IS NULL 
        AND video_url IS NOT NULL 
        AND video_url LIKE '%heygen%'
    `;
    console.log('  Backup complete.');
  } catch (err) {
    console.warn('  Backup failed:', (err as Error).message);
  }

  // Process each upload
  console.log('\nUpdating video_url to Blob URLs...');

  for (const upload of manifest.uploads) {
    try {
      // Map the download age category back to DB values
      // Download used: adult, kid, senior
      // DB uses: adult, child, teen, middleAge, senior, kid
      const ageVariants = getAgeVariants(upload.ageCategory);
      
      // Update all matching rows for this day+phase+age
      const result = await sql`
        UPDATE heygen_videos 
        SET video_url = ${upload.blobUrl}
        WHERE day_of_year = ${upload.day}
          AND phase = ${upload.phase}
          AND (age_category = ${ageVariants[0]} 
               OR age_category = ${ageVariants[1]} 
               OR age_category = ${ageVariants[2]})
          AND video_url LIKE '%heygen%'
        RETURNING id, day_of_year, phase, age_category
      `;

      if (result.length > 0) {
        updated += result.length;
        if (updated % 50 === 0) {
          process.stdout.write(`\r  Updated: ${updated} rows...      `);
        }
      } else {
        skipped++;
      }
    } catch (err) {
      errors++;
      console.error(`\n  Error updating Day ${upload.day} ${upload.phase}: ${(err as Error).message}`);
    }
  }

  console.log(`\n\nUpdate summary:`);
  console.log(`  Rows updated: ${updated}`);
  console.log(`  Skipped (no match or already updated): ${skipped}`);
  console.log(`  Errors: ${errors}`);

  // Verify: How many rows still have HeyGen URLs?
  console.log('\nVerification...');
  const remaining = await sql`
    SELECT COUNT(*)::int as count 
    FROM heygen_videos 
    WHERE video_url LIKE '%heygen%'
  `;
  const blobCount = await sql`
    SELECT COUNT(*)::int as count 
    FROM heygen_videos 
    WHERE video_url LIKE '%blob.vercel%'
  `;
  const totalWithUrl = await sql`
    SELECT COUNT(*)::int as count 
    FROM heygen_videos 
    WHERE video_url IS NOT NULL
  `;

  console.log(`  Total rows with video_url: ${totalWithUrl[0]?.count}`);
  console.log(`  Pointing to Vercel Blob:   ${blobCount[0]?.count}`);
  console.log(`  Still pointing to HeyGen:  ${remaining[0]?.count}`);

  if (remaining[0]?.count > 0) {
    console.log('\n  ⚠️ Some rows still point to HeyGen. Checking which...');
    const remainingDays = await sql`
      SELECT day_of_year, phase, age_category, video_url 
      FROM heygen_videos 
      WHERE video_url LIKE '%heygen%'
      ORDER BY day_of_year, phase
      LIMIT 20
    `;
    for (const row of remainingDays) {
      console.log(`    Day ${row.day_of_year} ${row.phase} (${row.age_category})`);
    }
  }

  // Quick liveness check on new Blob URLs
  console.log('\nLiveness check on new Blob URLs...');
  const blobSample = await sql`
    SELECT day_of_year, phase, video_url 
    FROM heygen_videos 
    WHERE video_url LIKE '%blob.vercel%'
    ORDER BY RANDOM()
    LIMIT 5
  `;
  for (const row of blobSample) {
    try {
      const resp = await fetch(row.video_url as string, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
      const sizeMB = (parseInt(resp.headers.get('content-length') || '0') / (1024 * 1024)).toFixed(1);
      console.log(`  Day ${row.day_of_year} ${row.phase}: ${resp.status} (${sizeMB}MB)`);
    } catch (err) {
      console.log(`  Day ${row.day_of_year} ${row.phase}: ERROR - ${(err as Error).message}`);
    }
  }

  console.log('\n=== DATABASE UPDATE COMPLETE ===');
}

function getAgeVariants(downloadAge: string): string[] {
  const mapping: Record<string, string[]> = {
    'adult': ['adult', 'middleAge', 'youngAdult'],
    'kid': ['kid', 'child', 'teen'],
    'senior': ['senior', 'elder', 'mature'],
  };
  return mapping[downloadAge] || [downloadAge, downloadAge, downloadAge];
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
