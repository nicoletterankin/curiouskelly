#!/usr/bin/env npx tsx
/**
 * Migration Script: Supabase Storage → Vercel Blob Storage
 * 
 * Migrates all video, audio, and visual assets from Supabase Storage
 * to Vercel Blob Storage for edge-optimized CDN delivery.
 * 
 * Usage:
 *   npx tsx scripts/migrate-to-blob.ts --dry-run
 *   npx tsx scripts/migrate-to-blob.ts --day 1
 *   npx tsx scripts/migrate-to-blob.ts --all
 */

import 'dotenv/config';
import { put } from '@vercel/blob';
import { createClient } from '@supabase/supabase-js';
import { parseArgs } from 'util';

const args = parseArgs({
  options: {
    'dry-run': { type: 'boolean', default: false },
    'day': { type: 'string' },
    'all': { type: 'boolean', default: false },
    'type': { type: 'string' }, // 'video', 'audio', 'visual'
  },
});

async function migrateAssets() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    console.error('❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
    process.exit(1);
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  const isDryRun = args.values['dry-run'];
  
  console.log('🚀 Starting migration to Vercel Blob Storage...');
  if (isDryRun) {
    console.log('⚠️  DRY RUN MODE - No files will be migrated');
  }
  console.log('');
  
  // Migrate videos
  if (!args.values.type || args.values.type === 'video') {
    console.log('📹 Migrating videos...');
    await migrateVideos(supabase, isDryRun);
  }
  
  // Migrate audio
  if (!args.values.type || args.values.type === 'audio') {
    console.log('\n🎵 Migrating audio files...');
    await migrateAudio(supabase, isDryRun);
  }
  
  // Migrate visuals
  if (!args.values.type || args.values.type === 'visual') {
    console.log('\n🖼️  Migrating visuals...');
    await migrateVisuals(supabase, isDryRun);
  }
  
  console.log('\n✅ Migration complete!');
}

async function migrateVideos(supabase: any, isDryRun: boolean) {
  let query = supabase
    .from('kelly_video_assets')
    .select('*')
    .eq('asset_type', 'video')
    .not('public_url', 'is', null);
  
  if (args.values.day) {
    query = query.eq('day_number', parseInt(args.values.day));
  }
  
  const { data: videos, error } = await query;
  
  if (error) {
    console.error('❌ Error fetching videos:', error);
    return;
  }
  
  console.log(`   Found ${videos.length} videos to migrate`);
  
  for (const video of videos) {
    try {
      // Extract path from Supabase URL
      const supabaseUrl = video.public_url;
      const pathMatch = supabaseUrl.match(/\/storage\/v1\/object\/public\/([^?]+)/);
      
      if (!pathMatch) {
        console.warn(`   ⚠️  Skipping ${video.id}: Invalid URL format`);
        continue;
      }
      
      const storagePath = pathMatch[1];
      const [bucket, ...pathParts] = storagePath.split('/');
      const fileName = pathParts[pathParts.length - 1];
      
      // Construct Vercel Blob path
      const blobPath = `videos/day-${String(video.day_number).padStart(3, '0')}/${video.archetype}/${video.phase}.mp4`;
      
      if (isDryRun) {
        console.log(`   [DRY RUN] Would migrate: ${storagePath} → ${blobPath}`);
        continue;
      }
      
      // Download from Supabase Storage
      const { data: blob, error: downloadError } = await supabase.storage
        .from(bucket)
        .download(pathParts.join('/'));
      
      if (downloadError || !blob) {
        console.warn(`   ⚠️  Failed to download ${storagePath}: ${downloadError?.message}`);
        continue;
      }
      
      // Upload to Vercel Blob
      const buffer = Buffer.from(await blob.arrayBuffer());
      const vercelBlob = await put(blobPath, buffer, {
        access: 'public',
        addRandomSuffix: false,
        cacheControlMaxAge: 31536000, // 1 year
      });
      
      // Update database with new URL
      await supabase
        .from('kelly_video_assets')
        .update({ 
          vercel_blob_url: vercelBlob.url,
          updated_at: new Date().toISOString(),
        })
        .eq('id', video.id);
      
      console.log(`   ✅ Migrated: day-${video.day_number}/${video.archetype}/${video.phase}.mp4`);
      
    } catch (error: any) {
      console.error(`   ❌ Error migrating video ${video.id}:`, error.message);
    }
  }
}

async function migrateAudio(supabase: any, isDryRun: boolean) {
  // Similar logic for audio files
  // Check lesson_atoms table for audio_url fields
  let query = supabase
    .from('lesson_atoms')
    .select('day_number, archetype, phase, audio_url')
    .not('audio_url', 'is', null);
  
  if (args.values.day) {
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('id')
      .eq('day_number', parseInt(args.values.day))
      .single();
    
    if (lesson) {
      query = query.eq('core_lesson_id', lesson.id);
    }
  }
  
  const { data: atoms, error } = await query;
  
  if (error) {
    console.error('❌ Error fetching audio:', error);
    return;
  }
  
  console.log(`   Found ${atoms.length} audio files to migrate`);
  
  for (const atom of atoms) {
    if (!atom.audio_url) continue;
    
    try {
      const blobPath = `audio/day-${String(atom.day_number).padStart(3, '0')}/${atom.archetype}/${atom.phase}.mp3`;
      
      if (isDryRun) {
        console.log(`   [DRY RUN] Would migrate: ${atom.audio_url} → ${blobPath}`);
        continue;
      }
      
      // Download and upload logic similar to videos
      // (Implementation depends on where audio is stored)
      console.log(`   ⚠️  Audio migration not yet implemented for: ${atom.audio_url}`);
      
    } catch (error: any) {
      console.error(`   ❌ Error migrating audio:`, error.message);
    }
  }
}

async function migrateVisuals(supabase: any, isDryRun: boolean) {
  // Similar logic for visuals
  let query = supabase
    .from('lesson_atoms')
    .select('day_number, archetype, phase, visual_url')
    .not('visual_url', 'is', null);
  
  if (args.values.day) {
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('id')
      .eq('day_number', parseInt(args.values.day))
      .single();
    
    if (lesson) {
      query = query.eq('core_lesson_id', lesson.id);
    }
  }
  
  const { data: atoms, error } = await query;
  
  if (error) {
    console.error('❌ Error fetching visuals:', error);
    return;
  }
  
  console.log(`   Found ${atoms.length} visuals to migrate`);
  
  for (const atom of atoms) {
    if (!atom.visual_url) continue;
    
    try {
      const blobPath = `visuals/day-${String(atom.day_number).padStart(3, '0')}/${atom.phase}-infographic.png`;
      
      if (isDryRun) {
        console.log(`   [DRY RUN] Would migrate: ${atom.visual_url} → ${blobPath}`);
        continue;
      }
      
      // Download and upload logic similar to videos
      console.log(`   ⚠️  Visual migration not yet implemented for: ${atom.visual_url}`);
      
    } catch (error: any) {
      console.error(`   ❌ Error migrating visual:`, error.message);
    }
  }
}

// Run migration
migrateAssets().catch(console.error);

