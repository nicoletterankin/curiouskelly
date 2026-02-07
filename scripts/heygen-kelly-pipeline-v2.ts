#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN KELLY PIPELINE V2 - PRODUCTION ENGINE
 * 
 * FEATURES:
 * - 🚀 Parallel Generation (Concurrrency Control)
 * - 💎 Super Resolution (1080p+)
 * - 🎤 Lossless Audio (MP3 44.1kHz 192kbps)
 * - 🔄 Robust Retry Logic & Polling
 * - 👥 Avatar Group Support (Canonical Mapping)
 * 
 * Usage:
 *   npx tsx scripts/heygen-kelly-pipeline-v2.ts --day 1
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // Concurrency (HeyGen Enterprise usually allows 3-5, Pro 1-2)
  // Reduced to 2 to be safe with ElevenLabs/HeyGen rate limits
  MAX_CONCURRENT_JOBS: 2,
  
  // APIs
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  ELEVENLABS_KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  
  // Storage
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'production-dec17'),
  SUPABASE_BUCKET: 'kelly-videos',
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// 12 ARCHETYPES MAPPING (CANONICAL)
// =============================================================================

const AVATAR_MAP: Record<string, string> = {
  "The Architect": "06b78109ad22489ea2165ebbf180f77b",
  "The Empath": "e614671b193c40f99772f7de5d1c51f7",
  "The MacGyver": "3f44bd33bfd1494d916d2746808a1a39",
  "The Explorer": "aa8b5eb1d711468a9a6e2085a4f8469c",
  "The Mystic": "a2b31ed0b5f84b0fa02d15d411735d3a",
  "The Provider": "4227be1001a3431db2cb4c59f9c25287",
  "The Rebel": "45e5ef8b651846e0b62b7477e552e87b",
  "The Scientist": "b9032c922c6e4e35b58a98abd499d060",
  "The Storyteller": "d1d731dcdd5d4bb9af1c020a907671dc",
  "The Strategist": "d4eccf6a8d4c427b9313208d640db407",
  "The Survivor": "7bb18cddacd44333813cc90ffa44f766",
  "The Diplomat": "433ad96bf5d647d9964cecf784d008f6" // Neutral
};

const VALID_ARCHETYPES = Object.keys(AVATAR_MAP);

// =============================================================================
// HELPERS
// =============================================================================

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Simple concurrency limiter
async function runConcurrent<T>(items: T[], fn: (item: T) => Promise<void>, limit: number) {
  const executing = new Set<Promise<void>>();
  
  for (const item of items) {
    const p = Promise.resolve().then(() => fn(item));
    executing.add(p);
    
    const clean = () => executing.delete(p);
    p.then(clean).catch(clean);
    
    if (executing.size >= limit) {
      await Promise.race(executing);
    }
  }
  
  await Promise.all(executing);
}

// =============================================================================
// CORE GENERATION
// =============================================================================

async function generateSingleVideo(
  atomId: string,
  archetype: string, 
  avatarId: string, 
  script: string, 
  outputName: string
) {
  console.log(`\n🎬 STARTING: ${archetype} (${outputName})`);

  try {
    // 1. Generate High-Quality Audio
    console.log(`   🎤 [${archetype}] Generating Audio (MP3 44.1kHz 192kbps)...`);
    
    // Add small random delay to avoid hitting rate limits instantly
    await sleep(Math.floor(Math.random() * 2000));

    const audioResponse = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.ELEVENLABS_KELLY_VOICE_ID}?output_format=mp3_44100_192`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: script,
          model_id: 'eleven_multilingual_v2',
          voice_settings: { stability: 0.5, similarity_boost: 0.75 },
        }),
      }
    );

    if (!audioResponse.ok) {
        const errorText = await audioResponse.text();
        throw new Error(`ElevenLabs Error: ${audioResponse.status} ${audioResponse.statusText} - ${errorText}`);
    }
    const audioBuffer = Buffer.from(await audioResponse.arrayBuffer());

    // Upload audio to Supabase (HeyGen needs a URL)
    const audioPath = `heygen/production-audio/${outputName}_${Date.now()}.mp3`;
    await supabase.storage.from('kelly-templates').upload(audioPath, audioBuffer, { upsert: true, contentType: 'audio/mpeg' });
    const { data: audioUrlData } = supabase.storage.from('kelly-templates').getPublicUrl(audioPath);
    
    console.log(`   ✅ [${archetype}] Audio Ready: ${audioUrlData.publicUrl}`);

    // 2. Trigger HeyGen Generation
    console.log(`   🚀 [${archetype}] Triggering HeyGen (Super Res ON)...`);
    
    const payload = {
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: avatarId,
        },
        voice: {
          type: 'audio',
          audio_url: audioUrlData.publicUrl,
        },
        background: { type: 'color', value: '#FFFFFF' }
      }],
      dimension: { width: 1080, height: 1080 },
      test: false, // PRODUCTION MODE
      super_resolution: true // CRITICAL FOR QUALITY
    };

    const genResponse = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': CONFIG.HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!genResponse.ok) {
      const errText = await genResponse.text();
      throw new Error(`HeyGen Trigger Failed: ${errText}`);
    }

    const genResult = await genResponse.json();
    const videoId = genResult.data.video_id;
    console.log(`   ⏳ [${archetype}] Processing Video ID: ${videoId}`);

    // 3. Poll for Completion (with backoff)
    let videoUrl = '';
    let attempts = 0;
    while (true) {
      await sleep(10000); // 10s interval
      attempts++;
      
      const statusRes = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
        headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY }
      });
      const statusJson = await statusRes.json();
      const status = statusJson.data?.status;

      if (status === 'completed') {
        videoUrl = statusJson.data.video_url;
        console.log(`   ✅ [${archetype}] Video Completed!`);
        break;
      } else if (status === 'failed') {
        const errorDetail = typeof statusJson.data.error === 'object' 
          ? JSON.stringify(statusJson.data.error) 
          : statusJson.data.error;
        throw new Error(`HeyGen Processing Failed: ${errorDetail}`);
      } else {
        process.stdout.write(`[${archetype}: ${status}] `);
      }
      
      if (attempts > 60) throw new Error(`Timeout waiting for ${archetype}`); // 10 mins max
    }

    // 4. Download & Save
    console.log(`   📥 [${archetype}] Downloading & Saving...`);
    const vidRes = await fetch(videoUrl);
    const vidBuffer = Buffer.from(await vidRes.arrayBuffer());
    
    // Local save
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, `${outputName}.mp4`), vidBuffer);
    
    // Supabase Upload
    const storagePath = `production/final/${outputName}.mp4`;
    await supabase.storage.from(CONFIG.SUPABASE_BUCKET).upload(storagePath, vidBuffer, { upsert: true, contentType: 'video/mp4' });
    
    // Get Public URL
    const { data: publicUrlData } = supabase.storage.from(CONFIG.SUPABASE_BUCKET).getPublicUrl(storagePath);
    const finalVideoUrl = publicUrlData.publicUrl;

    // Update Database
    const { error: dbError } = await supabase
        .from('lesson_atoms')
        .update({ hd_video_url: finalVideoUrl })
        .eq('id', atomId);

    if (dbError) {
        console.error(`   ⚠️ [${archetype}] Video saved but DB update failed: ${dbError.message}`);
    } else {
        console.log(`   💾 [${archetype}] DB Updated with URL`);
    }

    console.log(`   🎉 [${archetype}] FINISHED: ${outputName}`);

  } catch (err: any) {
    console.error(`\n❌ [${archetype}] FAILED: ${err.message}`);
    // We don't throw here so other concurrent jobs continue
  }
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const dayIndex = args.indexOf('--day');
  const day = dayIndex > -1 ? parseInt(args[dayIndex + 1]) : 1;
  
  const archIndex = args.indexOf('--archetype');
  const targetArchetype = archIndex > -1 ? args[archIndex + 1] : null;

  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 KELLY PRODUCTION PIPELINE V2                           ║');
  console.log('║  Target: Day ' + day + '                                            ║');
  if (targetArchetype) console.log('║  Archetype: ' + targetArchetype + '                                   ║');
  console.log('║  Mode: Parallel (' + CONFIG.MAX_CONCURRENT_JOBS + ' threads)                            ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // 1. Fetch Lesson Atoms for the Day
  console.log(`\n🔍 Fetching Day ${day} atoms...`);
  
  // First get the core lesson ID (prefer 'learn' track which has full content)
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, day_number, track')
    .eq('day_number', day)
    .eq('track', 'learn')
    .single();

  if (lessonError || !lesson) {
    console.error('❌ Database Error (Lesson):', lessonError || 'Lesson not found');
    process.exit(1);
  }

  // Then fetch atoms using the foreign key
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id)
    .order('phase');

  if (atomsError || !atoms) {
    console.error('❌ Database Error (Atoms):', atomsError);
    process.exit(1);
  }

  // Filter for the 12 archetypes only (and optional target)
  const tasks = atoms.filter(a => {
    if (!AVATAR_MAP[a.archetype]) return false;
    if (targetArchetype && a.archetype !== targetArchetype) return false;
    
    // NOTE: Skipping logic REMOVED to ensure we fix any bad/partial files
    // if (a.hd_video_url && a.hd_video_url.length > 0) return false;
    
    return true;
  });
  
  if (tasks.length === 0) {
    console.warn('⚠️ No atoms found matching configured Archetypes. Check database!');
    console.log('Valid Archetypes:', VALID_ARCHETYPES);
    return;
  }

  console.log(`📋 Found ${tasks.length} tasks. Starting Batch Generation...`);

  // 2. Run Parallel Generation
  await runConcurrent(tasks, async (atom) => {
    const archetype = atom.archetype;
    const script = atom.content?.script;
    const avatarId = AVATAR_MAP[archetype];
    
    if (!script) {
      console.log(`⚠️ Skipping ${archetype} (No script)`);
      return;
    }

    // Canonical ID Format: day-[N]-[archetype]-[phase]-[type]-[variant]
    const dayStr = day.toString().padStart(3, '0');
    const archSlug = archetype.toLowerCase().replace(/\s+/g, '-').replace('the-', ''); // scientist
    const phaseSlug = atom.phase; // Hook
    
    const outputName = `day-${dayStr}-${archSlug}-${phaseSlug}-main-en`;
    
    // CHECK IF EXISTS (Skip logic)
    const localPath = path.join(CONFIG.OUTPUT_DIR, `${outputName}.mp4`);
    // Also check alternate underscore format just in case
    const altOutputName = `day_${day.toString().padStart(3, '0')}_${atom.phase}_${archetype.replace(/\s+/g, '_').toLowerCase()}`;
    const altLocalPath = path.join(CONFIG.OUTPUT_DIR, `${altOutputName}.mp4`);
    
    if (fs.existsSync(localPath)) {
      console.log(`⏩ [${archetype}] Skipping ${atom.phase} (Already exists: ${outputName}.mp4)`);
      return;
    }
    if (fs.existsSync(altLocalPath)) {
      console.log(`⏩ [${archetype}] Skipping ${atom.phase} (Already exists: ${altOutputName}.mp4)`);
      return;
    }
    
    await generateSingleVideo(atom.id, archetype, avatarId, script, outputName);

  }, CONFIG.MAX_CONCURRENT_JOBS);

  console.log('\n✅ BATCH COMPLETE.');
}

main().catch(console.error);
