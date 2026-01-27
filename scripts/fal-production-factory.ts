#!/usr/bin/env npx tsx
/**
 * 🏭 FAL.AI PRODUCTION FACTORY
 * 
 * Generates lip-synced Kelly videos using fal.ai (bypassing HeyGen)
 * Uses existing audio_url from kelly_lesson_assets + Kelly reference images
 * 
 * Usage:
 *   npx tsx scripts/fal-production-factory.ts --limit=1   # Test with 1
 *   npx tsx scripts/fal-production-factory.ts --limit=50  # Production batch
 */

import 'dotenv/config';
import { fal } from '@fal-ai/client';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const FAL_KEY = process.env.FAL_KEY!;

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// =============================================================================
// KELLY REFERENCE IMAGES (public URLs or local paths)
// =============================================================================

// Pre-uploaded Kelly images for different ages (from Supabase storage)
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';

const KELLY_IMAGES = {
  // Core expression poses (good for lipsync)
  kid: `${SUPABASE_URL}/storage/v1/object/public/images/kelly_v2/marketing/age_variants/kelly-age15-closeup-1x1.png`,
  adult: `${SUPABASE_URL}/storage/v1/object/public/images/kelly_v2/core/chair/kelly-chair-explaining.png`,
  senior: `${SUPABASE_URL}/storage/v1/object/public/images/kelly_v2/core/chair/kelly-chair-wisdom.png`,
  // Backup/alternate
  adult_front: `${SUPABASE_URL}/storage/v1/object/public/images/kelly_v2/junk_drawer/kelly-expression-front-studio-neutral.png`,
  // Local fallback
  local: 'digital-kelly/assets/images/kelly_front.png',
};

function getKellyImage(ageGroup: number): string {
  if (ageGroup <= 12) return KELLY_IMAGES.kid;
  if (ageGroup >= 55) return KELLY_IMAGES.senior;
  return KELLY_IMAGES.adult;
}

interface LessonAsset {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  language: string;
  audio_url: string;
}

// =============================================================================
// FAL.AI LIPSYNC MODELS (in preference order)
// =============================================================================

const LIPSYNC_MODELS = [
  'fal-ai/sadtalker',
  'fal-ai/sync-lipsync',
  'fal-ai/liveportrait',
  'fal-ai/latent-sync',
] as const;

// =============================================================================
// UPLOAD TO FAL.AI STORAGE
// =============================================================================

async function uploadToFal(url: string): Promise<string> {
  // If already a public URL, might work directly
  if (url.startsWith('https://')) {
    return url;
  }
  
  // For local files, upload to fal storage
  if (fs.existsSync(url)) {
    const buffer = fs.readFileSync(url);
    const blob = new Blob([buffer]);
    // @ts-ignore
    const falUrl = await fal.storage.upload(blob);
    return falUrl;
  }
  
  return url;
}

async function uploadImageToFal(imagePath: string): Promise<string> {
  console.log(`[Fal] Uploading image to fal storage...`);
  
  // Try to fetch from URL first
  if (imagePath.startsWith('https://')) {
    try {
      const response = await fetch(imagePath);
      if (response.ok) {
        const buffer = await response.arrayBuffer();
        const blob = new Blob([Buffer.from(buffer)], { type: 'image/png' });
        // @ts-ignore
        const falUrl = await fal.storage.upload(blob);
        console.log(`[Fal] Uploaded: ${falUrl.substring(0, 50)}...`);
        return falUrl;
      }
    } catch (e) {
      console.log(`[Fal] URL fetch failed, trying local...`);
    }
  }
  
  // Fallback to local file
  const localPath = KELLY_IMAGES.local;
  if (fs.existsSync(localPath)) {
    const buffer = fs.readFileSync(localPath);
    const blob = new Blob([buffer], { type: 'image/png' });
    // @ts-ignore
    const falUrl = await fal.storage.upload(blob);
    console.log(`[Fal] Uploaded from local: ${falUrl.substring(0, 50)}...`);
    return falUrl;
  }
  
  throw new Error('No Kelly image available');
}

// =============================================================================
// GENERATE VIDEO WITH FAL.AI
// =============================================================================

async function generateVideo(imageUrl: string, audioUrl: string): Promise<{ url: string; model: string } | null> {
  console.log(`[Fal] Generating video...`);
  
  for (const model of LIPSYNC_MODELS) {
    console.log(`[Fal] Trying model: ${model}`);
    
    try {
      // @ts-ignore
      const result = await fal.subscribe(model, {
        input: {
          image_url: imageUrl,
          audio_url: audioUrl,
          source_image_url: imageUrl,
          driven_audio_url: audioUrl,
        },
        logs: false,
      });
      
      // Extract video URL from various possible locations
      const videoUrl = 
        result?.data?.video?.url ||
        result?.data?.video_url ||
        result?.video?.url ||
        result?.video_url ||
        (Array.isArray(result?.data) ? result.data[0]?.url : null);
      
      if (videoUrl) {
        console.log(`[Fal] ✅ ${model} succeeded!`);
        return { url: videoUrl, model: model.split('/')[1] };
      }
      
    } catch (error: any) {
      console.log(`[Fal] ❌ ${model}: ${error.message?.substring(0, 50) || 'Unknown error'}`);
    }
  }
  
  return null;
}

// =============================================================================
// DOWNLOAD AND UPLOAD TO SUPABASE
// =============================================================================

async function downloadAndUpload(
  videoUrl: string,
  dayNumber: number,
  phase: string,
  ageGroup: number
): Promise<string> {
  console.log(`[Upload] Downloading from fal...`);
  
  const response = await fetch(videoUrl);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  const fileName = `${phase}-age${ageGroup}-en.mp4`;
  const remotePath = `lipsync/day-${dayNumber}/${fileName}`;
  
  console.log(`[Upload] Uploading to Supabase: ${remotePath}`);
  
  const { error } = await supabase.storage
    .from('kelly-videos')
    .upload(remotePath, buffer, { contentType: 'video/mp4', upsert: true });
  
  if (error) {
    throw new Error(`Supabase upload failed: ${error.message}`);
  }
  
  const { data } = supabase.storage.from('kelly-videos').getPublicUrl(remotePath);
  return data.publicUrl;
}

// =============================================================================
// UPDATE DATABASE
// =============================================================================

async function updateRegistry(assetId: string, videoUrl: string, model: string): Promise<void> {
  const { error } = await supabase
    .from('kelly_lesson_assets')
    .update({
      video_url: videoUrl,
      video_source: 'fal',
      status: 'complete',
      updated_at: new Date().toISOString()
    })
    .eq('id', assetId);

  if (error) throw error;
  console.log(`[DB] Updated: ${assetId} → complete (${model})`);
}

// =============================================================================
// MAIN FACTORY LOOP
// =============================================================================

async function runFactory(limit: number = 10) {
  console.log('========================================');
  console.log('🏭 FAL.AI PRODUCTION FACTORY');
  console.log('   Bypassing HeyGen - Using fal.ai');
  console.log('========================================');

  if (!FAL_KEY) {
    console.error('❌ FAL_KEY not set in environment');
    console.log('   Get one from: https://fal.ai/dashboard/keys');
    process.exit(1);
  }
  
  console.log(`✅ FAL_KEY found: ${FAL_KEY.substring(0, 10)}...`);

  // Get audio_ready assets
  const { data: assets, error } = await supabase
    .from('kelly_lesson_assets')
    .select('*')
    .eq('status', 'audio_ready')
    .not('audio_url', 'is', null)
    .order('day_number', { ascending: true })
    .limit(limit);

  if (error) throw error;

  console.log(`\n📋 Found ${assets?.length || 0} assets to process`);

  if (!assets || assets.length === 0) {
    console.log('⚠️ No assets ready for video generation');
    return;
  }

  // List assets
  assets.forEach((a: LessonAsset) => {
    const ageLabel = a.age_group <= 12 ? 'kid' : a.age_group >= 55 ? 'senior' : 'adult';
    console.log(`   • Day ${a.day_number} | ${a.phase} | ${ageLabel} (${a.age_group})`);
  });

  // Upload Kelly image once (reuse for all)
  console.log('\n📸 Preparing Kelly reference image...');
  let kellyImageUrl: string;
  try {
    kellyImageUrl = await uploadImageToFal(KELLY_IMAGES.adult);
  } catch (e) {
    console.error('❌ Failed to upload Kelly image');
    process.exit(1);
  }

  let success = 0;
  let failed = 0;

  for (const asset of assets as LessonAsset[]) {
    try {
      console.log(`\n--- Day ${asset.day_number}/${asset.phase} age${asset.age_group} ---`);

      // Upload audio to fal storage
      console.log(`[Fal] Uploading audio...`);
      const audioResponse = await fetch(asset.audio_url);
      if (!audioResponse.ok) {
        throw new Error(`Audio fetch failed: ${audioResponse.status}`);
      }
      const audioBuffer = await audioResponse.arrayBuffer();
      const audioBlob = new Blob([Buffer.from(audioBuffer)], { type: 'audio/mpeg' });
      // @ts-ignore
      const audioUrl = await fal.storage.upload(audioBlob);

      // Generate video
      const result = await generateVideo(kellyImageUrl, audioUrl);
      
      if (!result) {
        throw new Error('All lipsync models failed');
      }

      // Download and upload to Supabase
      const videoUrl = await downloadAndUpload(
        result.url,
        asset.day_number,
        asset.phase,
        asset.age_group
      );

      // Update database
      await updateRegistry(asset.id, videoUrl, result.model);

      console.log(`✅ COMPLETE: ${videoUrl}`);
      success++;

      // Rate limit
      await new Promise(r => setTimeout(r, 2000));

    } catch (err: any) {
      console.error(`❌ FAILED: ${err.message}`);
      failed++;

      await supabase
        .from('kelly_lesson_assets')
        .update({ 
          status: 'error', 
          error_message: String(err.message || err),
          updated_at: new Date().toISOString()
        })
        .eq('id', asset.id);
    }
  }

  console.log('\n========================================');
  console.log(`🏭 FACTORY COMPLETE: ${success} success, ${failed} failed`);
  console.log('========================================');
}

// =============================================================================
// CLI
// =============================================================================

const args = process.argv.slice(2);
const limitArg = args.find(a => a.startsWith('--limit='));
const limit = limitArg ? parseInt(limitArg.split('=')[1]) : 10;

runFactory(limit).catch(console.error);
