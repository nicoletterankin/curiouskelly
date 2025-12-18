#!/usr/bin/env npx tsx
/**
 * 📦 BACKUP HEYGEN VIDEOS TO SUPABASE
 * Downloads completed HeyGen videos and uploads to permanent storage
 */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface ManifestVideo {
  video_id: string;
  status: string;
  video_url?: string;
  archived_url?: string;
}

async function downloadVideo(url: string): Promise<Buffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}

async function uploadToSupabase(buffer: Buffer, filename: string): Promise<string> {
  const storagePath = `heygen-archive/${filename}`;
  
  const { error } = await supabase.storage
    .from('kelly-videos')
    .upload(storagePath, buffer, {
      contentType: 'video/mp4',
      upsert: true
    });
  
  if (error) throw error;
  
  const { data } = supabase.storage.from('kelly-videos').getPublicUrl(storagePath);
  return data.publicUrl;
}

async function backupDay351() {
  console.log('\n📦 BACKUP DAY 351 HEYGEN VIDEOS\n');
  
  const manifestPath = 'generated-videos/day-351-manifest.json';
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  let backed = 0;
  let skipped = 0;
  let failed = 0;
  
  for (const [archetype, video] of Object.entries(manifest.videos) as [string, ManifestVideo][]) {
    if (video.archived_url) {
      console.log(`  ⏭️  ${archetype}: Already archived`);
      skipped++;
      continue;
    }
    
    if (video.status !== 'completed' || !video.video_url) {
      console.log(`  ⏭️  ${archetype}: Not completed`);
      skipped++;
      continue;
    }
    
    try {
      console.log(`  📥 ${archetype}: Downloading...`);
      const buffer = await downloadVideo(video.video_url);
      
      const filename = `day-351-${archetype}-${video.video_id}.mp4`;
      console.log(`  📤 ${archetype}: Uploading to Supabase...`);
      const permanentUrl = await uploadToSupabase(buffer, filename);
      
      manifest.videos[archetype].archived_url = permanentUrl;
      console.log(`  ✅ ${archetype}: Archived`);
      backed++;
      
    } catch (error: any) {
      console.log(`  ❌ ${archetype}: ${error.message}`);
      failed++;
    }
  }
  
  // Save manifest with archived URLs
  manifest.last_backup = new Date().toISOString();
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log('\n════════════════════════════════════════');
  console.log(`Backed up: ${backed}`);
  console.log(`Skipped: ${skipped}`);
  console.log(`Failed: ${failed}`);
  console.log(`Manifest updated: ${manifestPath}`);
  console.log('════════════════════════════════════════\n');
}

async function main() {
  console.log('📦 HEYGEN VIDEO BACKUP');
  console.log('══════════════════════════════════════════════════');
  console.log('Backing up HeyGen videos to permanent Supabase storage');
  console.log('(HeyGen URLs expire in ~7 days)\n');
  
  await backupDay351();
  
  console.log('✅ Backup complete');
}

main().catch(console.error);
