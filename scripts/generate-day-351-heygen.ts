#!/usr/bin/env npx tsx
/**
 * DAY 351 HEYGEN VIDEO GENERATOR
 * 
 * Generates talking-head videos for Day 351 using HeyGen API.
 * Uses Kelly avatar images from Supabase storage.
 * 
 * Usage:
 *   npx tsx scripts/generate-day-351-heygen.ts
 *   npx tsx scripts/generate-day-351-heygen.ts --archetype=scientist --age=adult
 *   npx tsx scripts/generate-day-351-heygen.ts --dry-run
 *   npx tsx scripts/generate-day-351-heygen.ts --phase=hook
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// Kelly Avatar IDs per archetype/age from avatar-registry (pre-uploaded to HeyGen)
// Format: {archetype}_{age} -> HeyGen talking_photo_id
const KELLY_AVATAR_IDS: Record<string, string> = {
  // Scientist (all ages)
  'scientist_kid': 'fa4a6780e25a49699ee4f75cb1f03103',
  'scientist_teen': '2099c4a6d84b4795b3eac9e1342181e8',
  'scientist_adult': '4ac0e56fc4424805b70934399a10b084',
  'scientist_elder': 'f4919db9bd0340c2ae40c3f35dc1d030',
  'scientist_super_elder': '98178c87897e4421884b535b7864ba86',
  // Explorer (all ages)
  'explorer_kid': 'd4e960f7a3424d869877f3a951adfae7',
  'explorer_teen': '87abbcd3f963419897f5c2b118055c50',
  'explorer_adult': '28c1517303d7463eb643238db1bc1b4f',
  'explorer_elder': '85228e932d8a4818803a86a85834968c',
  'explorer_super_elder': 'e4ab0d4d1f1b4dc9b81a1076b018557f',
  // Rebel (all ages)
  'rebel_kid': '5cff601bfb344015a65ff46c6b8cd70a',
  'rebel_teen': '05d388c18b724341aba7412ae4335cc4',
  'rebel_adult': '43ed1075e09045598c8a9b5c01822295',
  'rebel_elder': 'c2073ea84f8b4f8bacfc223a1506a2d2',
  'rebel_super_elder': '48acf8e12f984598afdf361f913135bd',
  // Architect (all ages)
  'architect_kid': 'deaa213342944dc2bf671abe1442e316',
  'architect_teen': '6827da0589374d68ac2bca97bf265ba6',
  'architect_adult': '2e9227fbb359457d9a32fbf6f7793b60',
  'architect_elder': 'c70723da67d44f928e28d65a4f45a4bd',
  'architect_super_elder': '86bbb664016c474eb828c121d8b35680',
};

// Kelly avatar SOURCE images (for uploading to HeyGen as talking photos)
const KELLY_AVATAR_IMAGES: Record<string, string> = {
  'scientist_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_scientist_head.png',
  'explorer_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_explorer_head.png',
  'rebel_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_rebel_head.png',
  'architect_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_architect_head.png',
  'diplomat_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_diplomat_head.png',
  'empath_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_empath_head.png',
  'macgyver_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_macgyver_head.png',
  'mystic_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_mystic_head.png',
  'provider_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_provider_head.png',
  'storyteller_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_storyteller_head.png',
  'strategist_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_strategist_head.png',
  'survivor_adult': 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_survivor_head.png',
};

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'video', '351');

// ═══════════════════════════════════════════════════════════════════
// DAY 351 SCRIPTS (main teaching content per phase)
// ═══════════════════════════════════════════════════════════════════

const DAY_351_SCRIPTS: Record<string, string> = {
  hook: "Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful—they're practicing. Without moving a muscle. It's called visualization, and the science behind it might change how you think about learning itself.",
  
  cliff: "Here's where it gets interesting. When you vividly imagine doing something—really see it, feel it, experience it in your mind—your brain activates almost the same way as when you actually do it. The neurons fire. The pathways light up. But here's the question that puzzled scientists for years...",
  
  fact1: "When you imagine performing an action, your motor cortex—that's the part of your brain that controls movement—lights up almost identically to when you actually move. Brain scans show about 90% overlap. Ninety percent. Your brain literally cannot tell the difference between vividly imagining something and doing it. It's practicing either way.",
  
  fact2: "Let me tell you about a famous experiment. Researchers took people who had never played piano and divided them into three groups. Group one physically practiced a simple piece for five days. Group two only imagined practicing—same piece, same time, but never touched a key. Group three did nothing. After five days, they scanned everyone's brains. The results shocked the scientific community.",
  
  fact3: "This isn't just lab science. Elite performers have known this for decades. Olympic athletes spend up to 50% of their training time on mental rehearsal. Surgeons visualize entire procedures before making a single cut. Concert pianists play through pieces in their minds on the flight to performances. The key they all discovered: specificity. Vague daydreaming doesn't work. You need vivid, detailed, multi-sensory imagination.",
  
  wisdom: "Here's today's wisdom: Your imagination is a practice field. The mind that rehearses builds pathways the passive mind never develops. Every time you vividly imagine doing something, you're laying down the neural tracks that make it easier to do for real. This is one of the few truly free performance enhancers available to every human being.",
  
  outro: "That's today's lesson. Your brain is more trainable than you ever imagined—literally. Visualization isn't wishful thinking. It's cognitive rehearsal that primes your brain for performance. Tonight, give it a try. Close your eyes. Pick something you want to master. And practice it in the one gym that's always open—your mind."
};

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

function getArg(name: string): string | undefined {
  const arg = process.argv.find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : undefined;
}

function hasFlag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════════
// HEYGEN API
// ═══════════════════════════════════════════════════════════════════

interface HeyGenVoice {
  voice_id: string;
  name: string;
  language: string;
  gender: string;
}

async function getHeyGenVoices(): Promise<HeyGenVoice[]> {
  const response = await fetch('https://api.heygen.com/v2/voices', {
    headers: {
      'X-Api-Key': HEYGEN_API_KEY!,
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HeyGen voices API failed: ${response.status}`);
  }

  const data = await response.json();
  return data.data?.voices || [];
}

async function uploadTalkingPhoto(imageUrl: string): Promise<string> {
  // Download image first
  const imageResponse = await fetch(imageUrl);
  if (!imageResponse.ok) {
    throw new Error(`Failed to download image: ${imageUrl}`);
  }
  
  const imageBuffer = await imageResponse.arrayBuffer();
  const base64Image = Buffer.from(imageBuffer).toString('base64');
  
  // Upload to HeyGen
  const response = await fetch('https://api.heygen.com/v2/photo_avatar', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: `data:image/png;base64,${base64Image}`,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HeyGen upload failed: ${response.status} - ${error.slice(0, 200)}`);
  }

  const data = await response.json();
  return data.data?.talking_photo_id || data.data?.id;
}

async function generateVideo(params: {
  talkingPhotoId: string;
  script: string;
  voiceId: string;
}): Promise<string> {
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: params.talkingPhotoId,
        },
        voice: {
          type: 'text',
          input_text: params.script,
          voice_id: params.voiceId,
        },
      }],
      dimension: {
        width: 1280,
        height: 720,
      },
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HeyGen generate failed: ${response.status} - ${error.slice(0, 200)}`);
  }

  const data = await response.json();
  return data.data?.video_id;
}

async function getVideoStatus(videoId: string): Promise<{ status: string; video_url?: string }> {
  const response = await fetch(
    `https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`,
    {
      headers: {
        'X-Api-Key': HEYGEN_API_KEY!,
      },
    }
  );

  if (!response.ok) {
    throw new Error(`HeyGen status failed: ${response.status}`);
  }

  const data = await response.json();
  return {
    status: data.data?.status || 'unknown',
    video_url: data.data?.video_url,
  };
}

async function waitForVideo(videoId: string, maxWaitMs: number = 300000): Promise<string> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < maxWaitMs) {
    const status = await getVideoStatus(videoId);
    
    if (status.status === 'completed' && status.video_url) {
      return status.video_url;
    }
    
    if (status.status === 'failed') {
      throw new Error('Video generation failed');
    }
    
    console.log(`  ⏳ Status: ${status.status}...`);
    await sleep(10000); // Check every 10 seconds
  }
  
  throw new Error('Timeout waiting for video');
}

async function downloadVideo(url: string, outputPath: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download video: ${response.status}`);
  }
  
  const buffer = await response.arrayBuffer();
  fs.writeFileSync(outputPath, Buffer.from(buffer));
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const dryRun = hasFlag('dry-run');
  const archetypeFilter = getArg('archetype') || 'scientist';
  const ageFilter = getArg('age') || 'adult';
  const phaseFilter = getArg('phase');
  
  const avatarKey = `${archetypeFilter}_${ageFilter}`;
  
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 DAY 351 HEYGEN VIDEO GENERATOR                             ║');
  console.log('║  Generating talking-head videos for Kelly                      ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  
  if (!HEYGEN_API_KEY) {
    console.error('❌ HEYGEN_API_KEY not found in environment');
    process.exit(1);
  }
  
  console.log(`🎭 Avatar: ${avatarKey}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log('');
  
  // Get phases to generate
  let phases = Object.keys(DAY_351_SCRIPTS);
  if (phaseFilter) {
    phases = phases.filter(p => p === phaseFilter);
    console.log(`🔍 Filtering to phase: ${phaseFilter}`);
  }
  
  if (dryRun) {
    console.log('🔍 DRY RUN MODE - No videos will be generated');
    console.log('');
    for (const phase of phases) {
      console.log(`  ${phase}.mp4`);
      console.log(`    Script: ${DAY_351_SCRIPTS[phase].slice(0, 60)}...`);
      console.log('');
    }
    return;
  }
  
  // Ensure output directory exists
  const avatarOutputDir = path.join(OUTPUT_DIR, avatarKey);
  if (!fs.existsSync(avatarOutputDir)) {
    fs.mkdirSync(avatarOutputDir, { recursive: true });
    console.log(`📁 Created: ${avatarOutputDir}`);
  }
  
  // Step 1: Get talking photo ID (already uploaded to HeyGen)
  console.log('📸 Looking up avatar...');
  const talkingPhotoId = KELLY_AVATAR_IDS[avatarKey];
  
  if (!talkingPhotoId) {
    console.error(`❌ No HeyGen avatar ID for: ${avatarKey}`);
    console.error(`   Available: ${Object.keys(KELLY_AVATAR_IDS).join(', ')}`);
    process.exit(1);
  }
  
  console.log(`  ✅ Using talking_photo_id: ${talkingPhotoId}`);
  
  // Step 2: Get a voice
  console.log('🎤 Fetching voices...');
  const voices = await getHeyGenVoices();
  const englishFemale = voices.find(v => 
    v.language?.toLowerCase().includes('english') && 
    v.gender?.toLowerCase() === 'female'
  ) || voices[0];
  
  console.log(`  Using voice: ${englishFemale.name} (${englishFemale.voice_id})`);
  
  // Step 3: Generate videos for each phase
  console.log('');
  console.log('🎬 Generating videos...');
  
  let success = 0;
  let failed = 0;
  
  for (let i = 0; i < phases.length; i++) {
    const phase = phases[i];
    const script = DAY_351_SCRIPTS[phase];
    const outputPath = path.join(avatarOutputDir, `${phase}.mp4`);
    
    console.log(`\n[${i + 1}/${phases.length}] ${phase}`);
    console.log(`  Script: ${script.slice(0, 50)}...`);
    
    try {
      // Generate
      console.log('  🎬 Requesting video...');
      const videoId = await generateVideo({
        talkingPhotoId,
        script,
        voiceId: englishFemale.voice_id,
      });
      console.log(`  📹 Video ID: ${videoId}`);
      
      // Wait for completion
      console.log('  ⏳ Waiting for render...');
      const videoUrl = await waitForVideo(videoId);
      
      // Download
      console.log('  📥 Downloading...');
      await downloadVideo(videoUrl, outputPath);
      
      const sizeMB = (fs.statSync(outputPath).size / 1024 / 1024).toFixed(2);
      console.log(`  ✅ Saved (${sizeMB} MB)`);
      success++;
      
    } catch (error) {
      console.error(`  ❌ Failed: ${error}`);
      failed++;
    }
  }
  
  console.log('');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`✅ Success: ${success}/${phases.length}`);
  if (failed > 0) {
    console.log(`❌ Failed: ${failed}`);
  }
  console.log(`📁 Output: ${avatarOutputDir}`);
  console.log('════════════════════════════════════════════════════════════════');
}

main().catch(console.error);
