#!/usr/bin/env npx tsx
/**
 * HEYGEN SMART SCENE GENERATOR
 * 
 * Generates videos that avoid the 10-second motion loop by:
 * 1. Splitting long scripts into ~8 second scenes
 * 2. Using different motion treatments per scene
 * 3. Cutting BEFORE the "loop seam" head shake at ~9-10 seconds
 * 
 * Usage:
 *   npx tsx scripts/heygen-smart-scene-generator.ts --day 351 --phase fact1
 *   npx tsx scripts/heygen-smart-scene-generator.ts --test
 */

import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

// ═══════════════════════════════════════════════════════════════════
// MOTION LIBRARY
// ═══════════════════════════════════════════════════════════════════

// These should be 3 different motion treatments of the SAME Kelly face
// Each created with a different motion prompt during Kling upscale
interface MotionLibrary {
  centered: string;    // Motion A: Minimal movement, calm authority
  listening: string;   // Motion B: Engaged, conversational
  expressive: string;  // Motion C: Dynamic, enthusiastic
}

// Placeholder - replace with actual avatar IDs after creating motion variants
const KELLY_MOTIONS: MotionLibrary = {
  centered: process.env.KELLY_MOTION_CENTERED || '7bb18cddacd44333813cc90ffa44f766',
  listening: process.env.KELLY_MOTION_LISTENING || 'a2b31ed0b5f84b0fa02d15d411735d3a', 
  expressive: process.env.KELLY_MOTION_EXPRESSIVE || '45e5ef8b651846e0b62b7477e552e87b',
};

const KELLY_VOICE_ID = '0015ce4f932b405b9fc3a5e2f5e92c46';

// Maximum scene duration to avoid the loop seam (head shake at ~9-10s)
const MAX_SCENE_SECONDS = 8;

// ═══════════════════════════════════════════════════════════════════
// SCRIPT SPLITTER
// ═══════════════════════════════════════════════════════════════════

interface ScriptSegment {
  text: string;
  motion: keyof MotionLibrary;
  estimatedDuration: number;
}

/**
 * Estimates speaking duration based on word count
 * Average speaking pace: ~150 words per minute = 2.5 words/second
 */
function estimateDuration(text: string): number {
  const words = text.split(/\s+/).filter(w => w.length > 0).length;
  return words / 2.5; // seconds
}

/**
 * Finds a natural break point in text (sentence end, comma, dash)
 * near the target position
 */
function findNaturalBreak(text: string, targetPosition: number): number {
  const breakChars = ['.', '!', '?', '—', ',', ';', ':'];
  
  // Search within 30% of target position
  const searchStart = Math.floor(targetPosition * 0.7);
  const searchEnd = Math.ceil(targetPosition * 1.3);
  
  let bestBreak = targetPosition;
  let bestDistance = Infinity;
  
  for (let i = searchStart; i < Math.min(searchEnd, text.length); i++) {
    if (breakChars.includes(text[i])) {
      const distance = Math.abs(i - targetPosition);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestBreak = i + 1; // Include the punctuation
      }
    }
  }
  
  // If no punctuation found, break at a space
  if (bestBreak === targetPosition) {
    for (let i = targetPosition; i < text.length; i++) {
      if (text[i] === ' ') {
        bestBreak = i;
        break;
      }
    }
  }
  
  return bestBreak;
}

/**
 * Splits a script into segments of ~8 seconds each,
 * alternating between motion treatments
 */
function splitScript(
  script: string, 
  phaseType: 'hook' | 'fact' | 'wisdom' | 'outro'
): ScriptSegment[] {
  const totalDuration = estimateDuration(script);
  
  // If short enough, single scene
  if (totalDuration <= MAX_SCENE_SECONDS) {
    const motion = phaseType === 'hook' || phaseType === 'outro' 
      ? 'expressive' 
      : phaseType === 'wisdom' 
        ? 'centered' 
        : 'listening';
    return [{ text: script, motion, estimatedDuration: totalDuration }];
  }
  
  // Calculate number of segments needed
  const numSegments = Math.ceil(totalDuration / MAX_SCENE_SECONDS);
  const segments: ScriptSegment[] = [];
  
  // Motion rotation based on phase type
  const motionRotation: (keyof MotionLibrary)[] = 
    phaseType === 'wisdom' 
      ? ['centered', 'listening', 'centered']
      : phaseType === 'hook' || phaseType === 'outro'
        ? ['expressive', 'listening', 'expressive']
        : ['listening', 'centered', 'listening'];
  
  let remaining = script;
  let segmentIndex = 0;
  
  while (remaining.length > 0 && segmentIndex < numSegments) {
    const targetLength = Math.floor(script.length / numSegments);
    
    if (segmentIndex === numSegments - 1) {
      // Last segment gets everything remaining
      segments.push({
        text: remaining.trim(),
        motion: motionRotation[segmentIndex % motionRotation.length],
        estimatedDuration: estimateDuration(remaining),
      });
      break;
    }
    
    const breakPoint = findNaturalBreak(remaining, targetLength);
    const segmentText = remaining.slice(0, breakPoint).trim();
    
    segments.push({
      text: segmentText,
      motion: motionRotation[segmentIndex % motionRotation.length],
      estimatedDuration: estimateDuration(segmentText),
    });
    
    remaining = remaining.slice(breakPoint).trim();
    segmentIndex++;
  }
  
  return segments;
}

// ═══════════════════════════════════════════════════════════════════
// HEYGEN API
// ═══════════════════════════════════════════════════════════════════

interface VideoInput {
  character: {
    type: 'talking_photo';
    talking_photo_id: string;
  };
  voice: {
    type: 'text';
    input_text: string;
    voice_id: string;
    speed: number;
  };
  background: {
    type: 'color';
    value: string;
  };
}

async function generateVideo(segments: ScriptSegment[], maxRetries = 3): Promise<string | null> {
  const videoInputs: VideoInput[] = segments.map(segment => ({
    character: {
      type: 'talking_photo',
      talking_photo_id: KELLY_MOTIONS[segment.motion],
    },
    voice: {
      type: 'text',
      input_text: segment.text,
      voice_id: KELLY_VOICE_ID,
      speed: 1.0,
    },
    background: {
      type: 'color',
      value: '#1a1a2e',
    },
  }));

  console.log(`\n📤 Generating video with ${segments.length} scene(s):`);
  segments.forEach((seg, i) => {
    console.log(`   Scene ${i + 1}: ${seg.motion} (~${seg.estimatedDuration.toFixed(1)}s)`);
    console.log(`           "${seg.text.slice(0, 50)}..."`);
  });

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    if (attempt > 1) {
      console.log(`\n🔄 Retry attempt ${attempt}/${maxRetries}...`);
      await new Promise(r => setTimeout(r, 5000)); // Wait 5s before retry
    }

    const response = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_inputs: videoInputs,
        dimension: { width: 1280, height: 720 },
      }),
    });

    const text = await response.text();
    
    // Handle non-JSON responses (error pages, rate limits)
    let data: any;
    try {
      data = JSON.parse(text);
    } catch (e) {
      console.error(`❌ Non-JSON response (${response.status}):`, text.slice(0, 100));
      if (attempt < maxRetries) continue;
      return null;
    }
    
    if (!response.ok) {
      console.error(`❌ Generation failed (${response.status}):`, data.error?.message || 'Unknown error');
      if (response.status >= 500 && attempt < maxRetries) continue;
      return null;
    }

    console.log('✅ Video job started:', data.data?.video_id);
    return data.data?.video_id;
  }

  return null;
}

async function pollForCompletion(videoId: string): Promise<string | null> {
  console.log('\n⏳ Waiting for render...');
  
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 10000));
    
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`,
      { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
    );
    
    const data = await response.json();
    const status = data.data?.status;
    
    process.stdout.write(`\r   Status: ${status}...`);
    
    if (status === 'completed') {
      console.log(' ✅');
      return data.data?.video_url;
    }
    
    if (status === 'failed') {
      console.log(' ❌');
      console.error('   Error:', data.data?.error);
      return null;
    }
  }
  
  return null;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN SMART SCENE GENERATOR                               ║');
  console.log('║  Avoiding the 10-second loop with 8-second scene cuts         ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  // Test mode - use a sample 16-second script
  if (process.argv.includes('--test')) {
    console.log('\n🧪 TEST MODE\n');
    
    const testScript = "The neuroscience: When you imagine performing an action, your motor cortex—the part that controls movement—lights up almost identically to when you actually move. Brain scans show about 90% overlap. Your brain literally can't tell the difference.";
    
    console.log('📝 Test script (16s estimated):');
    console.log(`   "${testScript}"\n`);
    
    const segments = splitScript(testScript, 'fact');
    
    console.log('✂️  Split into segments:');
    segments.forEach((seg, i) => {
      console.log(`\n   Segment ${i + 1} (${seg.motion}, ~${seg.estimatedDuration.toFixed(1)}s):`);
      console.log(`   "${seg.text}"`);
    });
    
    if (process.argv.includes('--generate')) {
      const videoId = await generateVideo(segments);
      if (videoId) {
        const url = await pollForCompletion(videoId);
        if (url) {
          console.log('\n🎬 VIDEO READY:');
          console.log(url);
        }
      }
    } else {
      console.log('\n💡 Add --generate to actually create the video');
    }
    
    return;
  }
  
  console.log('\n📖 Usage:');
  console.log('   npx tsx scripts/heygen-smart-scene-generator.ts --test');
  console.log('   npx tsx scripts/heygen-smart-scene-generator.ts --test --generate');
}

main().catch(console.error);
