#!/usr/bin/env npx tsx
/**
 * HEYGEN MULTI-MOTION TEST
 * 
 * Tests whether we can create a single video that switches between
 * different avatar_ids (different base motion treatments) mid-video.
 * 
 * The goal: Avoid the uncanny valley of a single 10-second motion looping.
 * 
 * Usage:
 *   npx tsx scripts/heygen-multi-motion-test.ts
 *   npx tsx scripts/heygen-multi-motion-test.ts --dry-run
 */

import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// ═══════════════════════════════════════════════════════════════════
// TWO DIFFERENT KELLY AVATAR IDS (same face, different motion base)
// Replace these with actual IDs from your HeyGen account
// ═══════════════════════════════════════════════════════════════════

// These should ideally be the SAME Kelly face photo, but uploaded/processed 
// with DIFFERENT motion treatments (different Kling seeds or motion prompts)
// 
// For testing, we'll use two EXISTING avatars from your library.
// Even if they're different Kelly variants, this tests the scene-stitching mechanism.
// 
// From your kelly-talking-photos.json:
const KELLY_MOTION_A = process.env.KELLY_MOTION_A_ID || '7bb18cddacd44333813cc90ffa44f766';  // Index 1
const KELLY_MOTION_B = process.env.KELLY_MOTION_B_ID || 'a2b31ed0b5f84b0fa02d15d411735d3a';  // Index 2

// Test script - split into two parts for the two motions
const SCRIPT_PART_1 = "Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful.";
const SCRIPT_PART_2 = "They're practicing. Without moving a muscle. It's called visualization, and the science behind it might change how you think about learning.";

// ═══════════════════════════════════════════════════════════════════
// API HELPERS
// ═══════════════════════════════════════════════════════════════════

async function generateMultiSceneVideo(): Promise<string | null> {
  console.log('🎬 Generating multi-scene video...');
  console.log(`   Motion A: ${KELLY_MOTION_A}`);
  console.log(`   Motion B: ${KELLY_MOTION_B}`);
  
  const payload = {
    video_inputs: [
      // SCENE 1: First half of hook with Motion A
      {
        character: {
          type: 'talking_photo',
          talking_photo_id: KELLY_MOTION_A,
        },
        voice: {
          type: 'text',
          input_text: SCRIPT_PART_1,
          voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46', // Kelly voice (audio-kelly2.mp3)
          speed: 1.0,
        },
        background: {
          type: 'color',
          value: '#1a1a2e',  // Dark background
        },
      },
      // SCENE 2: Second half of hook with Motion B (DIFFERENT BASE MOTION!)
      {
        character: {
          type: 'talking_photo',
          talking_photo_id: KELLY_MOTION_B,
        },
        voice: {
          type: 'text',
          input_text: SCRIPT_PART_2,
          voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46', // Kelly voice
          speed: 1.0,
        },
        background: {
          type: 'color',
          value: '#1a1a2e',
        },
      },
    ],
    dimension: {
      width: 1280,
      height: 720,
    },
  };

  console.log('\n📤 Payload:');
  console.log(JSON.stringify(payload, null, 2));

  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  
  if (!response.ok) {
    console.error('❌ Generation failed:', response.status);
    console.error(JSON.stringify(data, null, 2));
    return null;
  }

  console.log('✅ Video job started:', data.data?.video_id);
  return data.data?.video_id;
}

async function pollForCompletion(videoId: string): Promise<string | null> {
  console.log('\n⏳ Polling for completion...');
  
  const maxAttempts = 60; // 10 minutes max
  
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 10000)); // 10s between polls
    
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`,
      {
        headers: { 'X-Api-Key': HEYGEN_API_KEY! },
      }
    );
    
    const data = await response.json();
    const status = data.data?.status;
    
    console.log(`   [${i + 1}/${maxAttempts}] Status: ${status}`);
    
    if (status === 'completed') {
      return data.data?.video_url;
    }
    
    if (status === 'failed') {
      console.error('❌ Video failed:', data.data?.error);
      return null;
    }
  }
  
  console.error('❌ Timeout waiting for video');
  return null;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🧪 HEYGEN MULTI-MOTION TEST                                   ║');
  console.log('║  Testing avatar switching mid-video to avoid motion loops     ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');

  if (!HEYGEN_API_KEY) {
    console.error('❌ HEYGEN_API_KEY not found in environment');
    process.exit(1);
  }

  // Default avatar IDs are now set - ready to test!

  if (process.argv.includes('--dry-run')) {
    console.log('🔍 DRY RUN - Not sending to API');
    return;
  }

  // Generate the video
  const videoId = await generateMultiSceneVideo();
  
  if (!videoId) {
    process.exit(1);
  }

  // Poll for completion
  const videoUrl = await pollForCompletion(videoId);
  
  if (videoUrl) {
    console.log('');
    console.log('════════════════════════════════════════════════════════════════');
    console.log('🎬 VIDEO READY!');
    console.log('');
    console.log(videoUrl);
    console.log('');
    console.log('👀 Watch for the MOTION CHANGE at the midpoint.');
    console.log('   If successful, you should see Kelly\'s movement pattern');
    console.log('   shift naturally without a jarring loop.');
    console.log('════════════════════════════════════════════════════════════════');
  }
}

main().catch(console.error);
