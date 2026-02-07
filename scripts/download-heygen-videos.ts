/**
 * Download existing HeyGen videos before URLs expire
 * 
 * HeyGen URLs at files.heygen.ai are temporary and may expire.
 * This script downloads them and stores in Vercel Blob for permanence.
 * 
 * Usage: npx tsx scripts/download-heygen-videos.ts
 */

import { put } from '@vercel/blob';
import { config } from 'dotenv';
import pg from 'pg';

config({ path: '.env.local' });
config({ path: '.env' });

const { Pool } = pg;

// Neon PostgreSQL connection (NOT Supabase!)
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || process.env.NEON_DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

interface VideoRecord {
  id: string;
  day_number: number;
  phase: string;
  age_group: string;
  video_url: string;
  archetype?: string;
}

async function fetchWithRetry(url: string, retries = 3): Promise<Response> {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      if (response.ok) return response;
      console.log(`  Retry ${i + 1}/${retries}: Status ${response.status}`);
    } catch (err: any) {
      console.log(`  Retry ${i + 1}/${retries}: ${err.message}`);
    }
    await new Promise(r => setTimeout(r, 2000 * (i + 1)));
  }
  throw new Error(`Failed after ${retries} retries`);
}

async function downloadHeyGenVideos() {
  console.log('=== HeyGen Video Download Script ===\n');
  
  // Check environment
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('ERROR: BLOB_READ_WRITE_TOKEN not set');
    process.exit(1);
  }
  
  const dbUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;
  if (!dbUrl) {
    console.error('ERROR: DATABASE_URL or NEON_DATABASE_URL not set');
    process.exit(1);
  }
  
  console.log('Database URL:', dbUrl.split('@')[1]?.split('/')[0] || 'connected');
  
  try {
    // Query videos with HeyGen URLs
    const query = `
      SELECT id, day_number, phase, age_group, video_url, archetype
      FROM kelly_lesson_assets 
      WHERE video_url LIKE '%files.heygen.ai%'
      ORDER BY day_number, phase
    `;
    
    const result = await pool.query(query);
    const videos: VideoRecord[] = result.rows;
    
    console.log(`\nFound ${videos.length} HeyGen videos to download\n`);
    
    if (videos.length === 0) {
      console.log('No HeyGen videos found. Checking for any videos...');
      const checkQuery = `SELECT COUNT(*) as total, COUNT(video_url) as with_video FROM kelly_lesson_assets`;
      const checkResult = await pool.query(checkQuery);
      console.log('Total records:', checkResult.rows[0].total);
      console.log('Records with video_url:', checkResult.rows[0].with_video);
      return;
    }
    
    let downloaded = 0;
    let failed = 0;
    const failures: string[] = [];
    
    for (const video of videos) {
      const label = `Day ${video.day_number} / ${video.phase} / ${video.age_group}`;
      console.log(`Processing: ${label}`);
      
      try {
        // Download from HeyGen
        console.log(`  Downloading from HeyGen...`);
        const response = await fetchWithRetry(video.video_url);
        const arrayBuffer = await response.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: 'video/mp4' });
        
        console.log(`  Downloaded: ${(arrayBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);
        
        // Validate size
        if (arrayBuffer.byteLength < 100000) {
          throw new Error(`Video too small: ${arrayBuffer.byteLength} bytes`);
        }
        
        // Upload to Vercel Blob
        const dayPadded = String(video.day_number).padStart(3, '0');
        const filename = `video/kelly/day-${dayPadded}/${video.phase}-${video.age_group}.mp4`;
        
        console.log(`  Uploading to Blob: ${filename}`);
        const uploaded = await put(filename, blob, { 
          access: 'public',
          contentType: 'video/mp4'
        });
        
        console.log(`  Uploaded: ${uploaded.url}`);
        
        // Update database with new URL
        const updateQuery = `
          UPDATE kelly_lesson_assets 
          SET video_url = $1, updated_at = NOW()
          WHERE id = $2
        `;
        await pool.query(updateQuery, [uploaded.url, video.id]);
        
        console.log(`  ✓ Database updated\n`);
        downloaded++;
        
      } catch (err: any) {
        console.error(`  ✗ Failed: ${err.message}\n`);
        failed++;
        failures.push(`${label}: ${err.message}`);
      }
      
      // Rate limit
      await new Promise(r => setTimeout(r, 1000));
    }
    
    console.log('\n=== Summary ===');
    console.log(`Downloaded: ${downloaded}`);
    console.log(`Failed: ${failed}`);
    
    if (failures.length > 0) {
      console.log('\nFailures:');
      failures.forEach(f => console.log(`  - ${f}`));
    }
    
  } catch (err: any) {
    console.error('Database error:', err.message);
  } finally {
    await pool.end();
  }
}

// Also check what videos exist
async function auditVideos() {
  console.log('\n=== Video Audit ===\n');
  
  try {
    // Count by source
    const sourceQuery = `
      SELECT 
        CASE 
          WHEN video_url LIKE '%files.heygen.ai%' THEN 'heygen_direct'
          WHEN video_url LIKE '%blob.vercel-storage.com%' THEN 'vercel_blob'
          WHEN video_url IS NOT NULL THEN 'other'
          ELSE 'no_video'
        END as source,
        COUNT(*) as count
      FROM kelly_lesson_assets
      GROUP BY source
      ORDER BY count DESC
    `;
    
    const sourceResult = await pool.query(sourceQuery);
    console.log('Video sources:');
    sourceResult.rows.forEach((row: any) => {
      console.log(`  ${row.source}: ${row.count}`);
    });
    
    // Audio coverage
    const audioQuery = `
      SELECT 
        COUNT(*) as total,
        COUNT(audio_url) as with_audio
      FROM kelly_lesson_assets
    `;
    const audioResult = await pool.query(audioQuery);
    console.log(`\nAudio coverage: ${audioResult.rows[0].with_audio} / ${audioResult.rows[0].total}`);
    
    // Days with video
    const daysQuery = `
      SELECT DISTINCT day_number
      FROM kelly_lesson_assets
      WHERE video_url IS NOT NULL
      ORDER BY day_number
    `;
    const daysResult = await pool.query(daysQuery);
    console.log(`\nDays with video: ${daysResult.rows.map((r: any) => r.day_number).join(', ')}`);
    
  } catch (err: any) {
    console.error('Audit error:', err.message);
  }
}

async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--audit')) {
    await auditVideos();
  } else if (args.includes('--download')) {
    await downloadHeyGenVideos();
  } else {
    console.log('Usage:');
    console.log('  npx tsx scripts/download-heygen-videos.ts --audit     # Check current state');
    console.log('  npx tsx scripts/download-heygen-videos.ts --download  # Download HeyGen videos');
    
    // Default: run audit
    await auditVideos();
  }
  
  await pool.end();
}

main().catch(console.error);
