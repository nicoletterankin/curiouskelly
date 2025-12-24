#!/usr/bin/env npx tsx
/**
 * 🚨 EMERGENCY: Generate Day 354 videos using SadTalker via fal.ai
 * 
 * Since HeyGen is stuck, we use SadTalker as fallback.
 * 
 * Usage: npx tsx scripts/emergency-day354-sadtalker.ts
 */

import 'dotenv/config';
import { fal } from '@fal-ai/client';
import { createClient } from '@supabase/supabase-js';

// Configure fal.ai
fal.config({ credentials: process.env.FAL_KEY! });

// Supabase client
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!;
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Kelly reference image (verified working photorealistic image)
const KELLY_IMAGE = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/kelly_1765361262640.png';

// Day 354 audio files (most recent versions)
const DAY_354_AUDIO: Record<string, string> = {
  hook: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_hook_1766200432881.mp3',
  cliff: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_cliff_1766200437014.mp3',
  q1: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_q1_1766200441386.mp3',
  q2: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_q2_1766200447085.mp3',
  q3: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_q3_1766200453090.mp3',
  wisdom: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_wisdom_1766200458293.mp3',
  outro: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/audio/day_354_outro_1766200460705.mp3',
};

const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

async function generateVideo(phase: string, audioUrl: string): Promise<string | null> {
  console.log(`\n📽️  Phase: ${phase}`);
  console.log(`   Audio: ${audioUrl.substring(audioUrl.lastIndexOf('/') + 1)}`);
  console.log('   ⏳ Processing with SadTalker...');
  
  try {
    const result = await fal.subscribe('fal-ai/sadtalker', {
      input: {
        source_image_url: KELLY_IMAGE,
        driven_audio_url: audioUrl,
        still: false, // Allow head motion
        enhancer: 'gfpgan', // Face enhancement
        preprocess: 'crop', // Crop to face
      },
      logs: false,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') {
          process.stdout.write('.');
        }
      }
    });
    
    const videoUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
    
    if (!videoUrl) {
      console.log('\n   ❌ No video URL in response');
      return null;
    }
    
    console.log('\n   ✅ Video generated!');
    
    // Download and upload to Supabase
    console.log('   📤 Uploading to Supabase...');
    const response = await fetch(videoUrl);
    const buffer = Buffer.from(await response.arrayBuffer());
    
    const storagePath = `production/day_354/day_354_${phase}_explorer_sadtalker.mp4`;
    
    const { error: uploadError } = await supabase.storage
      .from('kelly-videos')
      .upload(storagePath, buffer, { 
        contentType: 'video/mp4', 
        upsert: true 
      });
    
    if (uploadError) {
      console.log(`   ⚠️  Upload error: ${uploadError.message}`);
      // Still return the fal.ai URL as backup
      return videoUrl;
    }
    
    const { data: urlData } = supabase.storage
      .from('kelly-videos')
      .getPublicUrl(storagePath);
    
    const publicUrl = urlData.publicUrl;
    console.log(`   📁 Uploaded: ${storagePath}`);
    
    // Register in kelly_video_assets
    await supabase.from('kelly_video_assets').upsert({
      lesson_day: 354,
      phase,
      template: 'The Explorer',
      age_bucket: 'adult',
      public_url: publicUrl,
      storage_path: storagePath,
      status: 'validated',
      quality_tier: 'standard',
      language: 'en',
    }, {
      onConflict: 'lesson_day,phase,template,age_bucket,language',
    });
    
    return publicUrl;
    
  } catch (error) {
    console.log(`\n   ❌ Error: ${(error as Error).message}`);
    return null;
  }
}

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║       🚨 EMERGENCY: DAY 354 SADTALKER GENERATION             ║
╚══════════════════════════════════════════════════════════════╝

Kelly Image: ${KELLY_IMAGE}
Phases: ${PHASES.length}
`);

  if (!process.env.FAL_KEY) {
    console.error('❌ FAL_KEY not set in environment');
    process.exit(1);
  }

  const results: Record<string, string | null> = {};
  let success = 0;
  let failed = 0;

  for (const phase of PHASES) {
    const audioUrl = DAY_354_AUDIO[phase];
    if (!audioUrl) {
      console.log(`\n⚠️  No audio for phase: ${phase}`);
      results[phase] = null;
      failed++;
      continue;
    }

    const videoUrl = await generateVideo(phase, audioUrl);
    results[phase] = videoUrl;
    
    if (videoUrl) {
      success++;
    } else {
      failed++;
    }
  }

  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                        📊 SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝

Success: ${success}/${PHASES.length}
Failed: ${failed}/${PHASES.length}

Videos:
`);

  for (const [phase, url] of Object.entries(results)) {
    const icon = url ? '✅' : '❌';
    console.log(`  ${icon} ${phase}: ${url ? 'Generated' : 'Failed'}`);
  }

  if (success > 0) {
    console.log(`
✨ Next steps:
   1. Test: http://localhost:3000/learn.html?day=354
   2. If working, approve: npx tsx scripts/approve-day.ts --day=354
`);
  }
}

main().catch(console.error);




