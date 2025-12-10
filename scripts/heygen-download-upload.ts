#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN VIDEO DOWNLOADER & UPLOADER
 * 
 * Downloads completed HeyGen videos and uploads to Supabase.
 * Updates lesson_atoms with the video URL.
 * 
 * Usage:
 *   npx tsx scripts/heygen-download-upload.ts <video_id> <day> <phase> <archetype>
 *   npx tsx scripts/heygen-download-upload.ts 720bfb769e924d3391c14dbdc60e13b6 1 Hook "The Scientist"
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'heygen-kelly'),
  BUCKET: 'kelly-videos',
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

async function getVideoStatus(videoId: string) {
  const response = await fetch(
    `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
    { headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY } }
  );
  return response.json();
}

async function waitForCompletion(videoId: string): Promise<string> {
  console.log(`⏳ Waiting for video ${videoId}...`);
  
  while (true) {
    const result = await getVideoStatus(videoId);
    process.stdout.write(`\r   Status: ${result.data.status}    `);
    
    if (result.data.status === 'completed') {
      console.log('\n✅ Complete!');
      return result.data.video_url;
    }
    
    if (result.data.status === 'failed') {
      throw new Error(`Video failed: ${result.data.error}`);
    }
    
    await new Promise(r => setTimeout(r, 10000));
  }
}

async function downloadVideo(url: string, outputName: string): Promise<Buffer> {
  console.log('📥 Downloading video...');
  const response = await fetch(url);
  const buffer = Buffer.from(await response.arrayBuffer());
  
  // Save locally
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  const localPath = path.join(CONFIG.OUTPUT_DIR, `${outputName}.mp4`);
  fs.writeFileSync(localPath, buffer);
  console.log(`   💾 Saved: ${localPath}`);
  
  return buffer;
}

async function uploadToSupabase(buffer: Buffer, outputName: string): Promise<string> {
  console.log('☁️ Uploading to Supabase...');
  
  const remotePath = `production/heygen/${outputName}.mp4`;
  
  const { error } = await supabase.storage
    .from(CONFIG.BUCKET)
    .upload(remotePath, buffer, {
      contentType: 'video/mp4',
      upsert: true,
    });

  if (error) throw error;

  const { data } = supabase.storage.from(CONFIG.BUCKET).getPublicUrl(remotePath);
  console.log(`   ✅ URL: ${data.publicUrl}`);
  
  return data.publicUrl;
}

async function updateDatabase(day: number, phase: string, archetype: string, videoUrl: string) {
  console.log('📝 Updating database...');
  
  const { error } = await supabase
    .from('lesson_atoms')
    .update({ hd_video_url: videoUrl })
    .eq('day_number', day)
    .eq('phase', phase)
    .eq('archetype', archetype);

  if (error) {
    console.error('   ⚠️ Database update failed:', error.message);
  } else {
    console.log(`   ✅ Updated: Day ${day}, ${phase}, ${archetype}`);
  }
}

async function main() {
  const [videoId, dayStr, phase, archetype] = process.argv.slice(2);

  if (!videoId) {
    console.log('Usage: npx tsx scripts/heygen-download-upload.ts <video_id> [day] [phase] [archetype]');
    console.log('Example: npx tsx scripts/heygen-download-upload.ts abc123 1 Hook "The Scientist"');
    process.exit(1);
  }

  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN VIDEO DOWNLOAD & UPLOAD                         ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log(`Video ID: ${videoId}`);

  try {
    // Wait for video
    const videoUrl = await waitForCompletion(videoId);

    // Create output name
    const day = parseInt(dayStr) || 0;
    const outputName = day && phase && archetype
      ? `day_${String(day).padStart(3, '0')}_${phase.toLowerCase()}_${archetype.replace(/\s+/g, '_').toLowerCase()}`
      : `heygen_${videoId}`;

    // Download
    const buffer = await downloadVideo(videoUrl, outputName);

    // Upload to Supabase
    const supabaseUrl = await uploadToSupabase(buffer, outputName);

    // Update database if we have lesson info
    if (day && phase && archetype) {
      await updateDatabase(day, phase, archetype, supabaseUrl);
    }

    console.log('\n' + '═'.repeat(60));
    console.log('🎉 COMPLETE!');
    console.log('═'.repeat(60));
    console.log(`Local: ${path.join(CONFIG.OUTPUT_DIR, outputName)}.mp4`);
    console.log(`Supabase: ${supabaseUrl}`);
    
    if (day) {
      console.log(`\nTest: http://localhost:3000/learn?day=${day}&clearcache=1`);
    }

  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
    process.exit(1);
  }
}

main();

