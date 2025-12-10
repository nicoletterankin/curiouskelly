#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN DAY 1 BATCH VIDEO GENERATOR
 * 
 * Generates all Day 1 lesson videos using HeyGen Photo Avatars.
 * Uses your 12 Kelly archetype avatars with archetype-specific motion prompts.
 * 
 * PREREQUISITES:
 * 1. Upload 12 Kelly images to HeyGen as Photo Avatars
 * 2. Update KELLY_AVATAR_IDS below with the talking_photo_id for each
 * 3. Ensure ElevenLabs audio is generated for Day 1
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  ELEVENLABS_KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// KELLY AVATAR IDS - UPDATE THESE AFTER UPLOADING TO HEYGEN
// =============================================================================

const KELLY_AVATAR_IDS: Record<string, string> = {
  // Format: "archetype": "talking_photo_id_from_heygen"
  "The Scientist": "PASTE_SCIENTIST_AVATAR_ID_HERE",
  "The Explorer": "PASTE_EXPLORER_AVATAR_ID_HERE",
  "The Rebel": "PASTE_REBEL_AVATAR_ID_HERE",
  "The Architect": "PASTE_ARCHITECT_AVATAR_ID_HERE",
  "The Diplomat": "PASTE_DIPLOMAT_AVATAR_ID_HERE",
  "The Empath": "PASTE_EMPATH_AVATAR_ID_HERE",
  "The MacGyver": "PASTE_MACGYVER_AVATAR_ID_HERE",
  "The Mystic": "PASTE_MYSTIC_AVATAR_ID_HERE",
  "The Provider": "PASTE_PROVIDER_AVATAR_ID_HERE",
  "The Storyteller": "PASTE_STORYTELLER_AVATAR_ID_HERE",
  "The Strategist": "PASTE_STRATEGIST_AVATAR_ID_HERE",
  "The Survivor": "PASTE_SURVIVOR_AVATAR_ID_HERE",
};

// =============================================================================
// MOTION PROMPTS - ARCHETYPE-SPECIFIC
// =============================================================================

const MOTION_PROMPTS: Record<string, string> = {
  "The Scientist": "Measured confidence, deliberate hand gestures presenting data. Thoughtful pauses, steady focused eye contact. Eyebrow raises on key points. Intellectual steepled hands. Minimal movement. Camera remains static.",
  "The Explorer": "Infectious enthusiasm, eyes wide with wonder. Animated pointing outward at discoveries. Curious head tilts. Genuine smile reaching eyes. Adventurous forward lean. Camera remains static.",
  "The Rebel": "Bold challenging energy. Unwavering eye contact. Knowing smirk. Provocative head tilt. Assertive hand emphasis. Chin down, looking through eyebrows. Edgy but approachable. Camera remains static.",
  "The Architect": "Calm methodical precision. Structured organized gestures building concepts. Steady centered posture. Deep concentration. Deliberate purposeful movement. Never rushed. Camera remains static.",
  "The Diplomat": "Warm inclusive energy. Open welcoming gestures. Gentle understanding nods. Soft approachable eye contact. Slight interested lean. Balanced fair movements. Camera remains static.",
  "The Empath": "Gentle nurturing warmth. Soft caring eye contact. Hand touches heart for emotional connection. Compassionate head tilt. Soft flowing movements. Leaning in creating safe space. Camera remains static.",
  "The MacGyver": "Practical hands-on energy. Active gestures showing how things work. Eyes bright with problem-solving. Can-do confidence nods. Action-ready forward lean. Resourceful energy. Camera remains static.",
  "The Mystic": "Serene profound calm. Slow graceful movements. Eyes with depth and ancient knowing. Slight upward gaze accessing insight. Peaceful knowing smile. Maximum presence, minimal movement. Camera remains static.",
  "The Provider": "Steady reassuring strength. Grounded reliable posture. Protective warm eye contact. Hand on heart or encompassing safety gestures. Stable dependable movements. Encouraging nods. Camera remains static.",
  "The Storyteller": "Theatrical captivating animation. Eyes sparkle with narrative magic. Expressive gestures painting pictures. Dynamic facial expressions. Dramatic pauses. Conspiratorial lean when sharing secrets. Camera remains static.",
  "The Strategist": "Sharp tactical confidence. Precise chess-piece hand gestures. Calculated intelligent eyes. Knowing winner's smile. Commanding head position. Decisive purposeful movements. Camera remains static.",
  "The Survivor": "Grounded no-nonsense directness. Unwavering resilient eye contact. Minimal meaningful gestures. Strong experienced posture. Serious determination. Practical battle-tested energy. Camera remains static.",
};

// =============================================================================
// AUDIO GENERATION (ElevenLabs)
// =============================================================================

async function generateAudio(script: string): Promise<string> {
  console.log('   🎤 Generating audio with ElevenLabs...');
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.ELEVENLABS_KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  
  // Upload to Supabase for HeyGen to access
  const fileName = `audio_${Date.now()}.mp3`;
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(`heygen/audio/${fileName}`, audioBuffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });

  if (error) throw error;

  const { data } = supabase.storage
    .from('kelly-templates')
    .getPublicUrl(`heygen/audio/${fileName}`);

  return data.publicUrl;
}

// =============================================================================
// HEYGEN VIDEO GENERATION
// =============================================================================

async function generateHeyGenVideo(
  avatarId: string,
  audioUrl: string,
  motionPrompt: string
): Promise<string> {
  console.log('   🎬 Generating HeyGen video...');
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': CONFIG.HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [
        {
          character: {
            type: 'talking_photo',
            talking_photo_id: avatarId,
          },
          voice: {
            type: 'audio',
            audio_url: audioUrl,
          },
          background: {
            type: 'color',
            value: '#FFFFFF',
          },
        },
      ],
      dimension: {
        width: 1920,
        height: 1080,
      },
      // Custom motion prompt
      ...(motionPrompt && { motion_prompt: motionPrompt }),
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HeyGen error: ${error}`);
  }

  const result = await response.json();
  return result.data.video_id;
}

async function waitForVideo(videoId: string): Promise<string> {
  console.log(`   ⏳ Waiting for video ${videoId}...`);
  
  while (true) {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      {
        headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY },
      }
    );

    const result = await response.json();
    
    if (result.data.status === 'completed') {
      return result.data.video_url;
    }
    
    if (result.data.status === 'failed') {
      throw new Error(`Video generation failed: ${result.data.error}`);
    }

    await new Promise(r => setTimeout(r, 15000)); // Check every 15 seconds
  }
}

// =============================================================================
// MAIN BATCH PROCESS
// =============================================================================

async function processDay1() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN DAY 1 BATCH VIDEO GENERATOR                     ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // Verify avatar IDs are set
  const missingAvatars = Object.entries(KELLY_AVATAR_IDS)
    .filter(([_, id]) => id.includes('PASTE_'))
    .map(([name]) => name);

  if (missingAvatars.length > 0) {
    console.error('\n❌ Missing avatar IDs! Update KELLY_AVATAR_IDS in this script:');
    missingAvatars.forEach(a => console.error(`   - ${a}`));
    console.error('\n1. Upload Kelly images to HeyGen: app.heygen.com → Avatars');
    console.error('2. Copy the talking_photo_id for each');
    console.error('3. Paste into KELLY_AVATAR_IDS above');
    process.exit(1);
  }

  // Get Day 1 lesson atoms
  const { data: atoms, error } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('day_number', 1)
    .order('phase');

  if (error || !atoms?.length) {
    console.error('❌ Could not fetch Day 1 lesson atoms:', error);
    process.exit(1);
  }

  console.log(`\n📚 Found ${atoms.length} lesson atoms for Day 1`);

  // Process each archetype
  const archetypes = ['The Scientist', 'The Explorer', 'The Rebel'];
  const phases = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];

  const results: Array<{ archetype: string; phase: string; videoUrl?: string; error?: string }> = [];

  for (const archetype of archetypes) {
    const avatarId = KELLY_AVATAR_IDS[archetype];
    const motionPrompt = MOTION_PROMPTS[archetype];

    console.log(`\n🎭 Processing archetype: ${archetype}`);

    for (const phase of phases) {
      const atom = atoms.find(a => a.archetype === archetype && a.phase === phase);
      
      if (!atom) {
        console.log(`   ⚠️ No atom found for ${archetype} - ${phase}`);
        results.push({ archetype, phase, error: 'No atom found' });
        continue;
      }

      const script = atom.content?.script;
      if (!script) {
        console.log(`   ⚠️ No script for ${archetype} - ${phase}`);
        results.push({ archetype, phase, error: 'No script' });
        continue;
      }

      console.log(`\n📝 ${archetype} - ${phase}`);
      console.log(`   Script: "${script.substring(0, 50)}..."`);

      try {
        // 1. Generate audio
        const audioUrl = await generateAudio(script);
        console.log(`   ✅ Audio: ${audioUrl.substring(0, 50)}...`);

        // 2. Generate video
        const videoId = await generateHeyGenVideo(avatarId, audioUrl, motionPrompt);
        console.log(`   🎬 Video ID: ${videoId}`);

        // 3. Wait for completion
        const videoUrl = await waitForVideo(videoId);
        console.log(`   ✅ Video complete!`);

        // 4. Update database
        await supabase
          .from('lesson_atoms')
          .update({ hd_video_url: videoUrl })
          .eq('id', atom.id);

        results.push({ archetype, phase, videoUrl });

      } catch (err: any) {
        console.error(`   ❌ Error: ${err.message}`);
        results.push({ archetype, phase, error: err.message });
      }
    }
  }

  // Summary
  console.log('\n\n' + '═'.repeat(60));
  console.log('📋 RESULTS');
  console.log('═'.repeat(60));

  const successful = results.filter(r => r.videoUrl);
  const failed = results.filter(r => r.error);

  console.log(`✅ Successful: ${successful.length}`);
  console.log(`❌ Failed: ${failed.length}`);

  if (failed.length > 0) {
    console.log('\nFailed items:');
    failed.forEach(f => console.log(`   - ${f.archetype} / ${f.phase}: ${f.error}`));
  }

  console.log('\n🎯 Day 1 processing complete!');
  console.log('Test at: http://localhost:3000/learn?day=1&clearcache=1');
}

processDay1().catch(console.error);

