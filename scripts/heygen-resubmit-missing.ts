#!/usr/bin/env npx tsx
/**
 * 🔧 HEYGEN RESUBMIT MISSING VIDEOS
 * Resubmits explorer, mystic, provider for Day 351
 */
import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const HEYGEN_API = 'https://api.heygen.com';

// Avatar IDs from motion library (motion B - main teaching)
const MISSING_ARCHETYPES = {
  explorer: '28c1517303d7463eb643238db1bc1b4f',
  mystic: '7729b75d6e204be48b577bcf70b46e9d', 
  provider: '4227be1001a3431db2cb4c59f9c25287'
};

// Day 351 script content
const LESSON_SCRIPT = `Visualization is mental practice—rehearsing success in your mind before doing it in reality. The brain doesn't fully distinguish imagined from real; mental practice builds real capacity. Olympic athletes use visualization as a core training technique. Mental rehearsal activates the same neural pathways as physical practice. Surgeons who visualize procedures before performing them have better outcomes. The quietest mind often hears the clearest answers.`;

// Phase scripts for multi-scene video
const PHASE_SCRIPTS = {
  hook: "Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful—they're practicing. Without moving a muscle.",
  cliff: "Here's where it gets interesting. When you vividly imagine doing something—really see it, feel it, experience it in your mind—your brain activates almost the same way as when you actually do it.",
  fact1: "Olympic athletes use visualization as a core training technique. Studies show that mental practice combined with physical practice outperforms physical practice alone.",
  fact2: "Mental rehearsal activates the same neural pathways as physical practice. Your brain literally can't tell the difference between vividly imagining an action and actually doing it.",
  fact3: "Surgeons who visualize procedures before performing them have better outcomes. Pilots, musicians, and public speakers use the same technique.",
  wisdom: "What you visualize, you can actualize. The mind is a rehearsal stage, and every vivid mental run-through builds real capacity.",
  outro: "That's your lesson for today. See you tomorrow for more ways to unlock your potential."
};

async function submitVideo(archetype: string, avatarId: string): Promise<{ video_id: string }> {
  console.log(`\n📹 Submitting ${archetype}...`);
  
  // Build multi-scene video
  const scenes = Object.entries(PHASE_SCRIPTS).map(([phase, script], index) => ({
    type: 'talking_photo',
    talking_photo_id: avatarId,
    voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46', // Kelly's HeyGen voice
    input_text: script,
    voice_type: 'elevenlabs',
    properties: {
      horizontal_align: 'center',
      scale: 1.0
    }
  }));

  const payload = {
    video_inputs: scenes.map(scene => ({
      character: {
        type: 'talking_photo',
        talking_photo_id: scene.talking_photo_id,
        talking_style: 'expressive'
      },
      voice: {
        type: 'text',
        input_text: scene.input_text,
        voice_id: scene.voice_id
      }
    })),
    dimension: { width: 1280, height: 720 },
    test: false
  };

  console.log(`  Scenes: ${scenes.length}`);
  
  const response = await fetch(`${HEYGEN_API}/v2/video/generate`, {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HeyGen API error ${response.status}: ${text}`);
  }

  const data = await response.json();
  console.log(`  ✅ Submitted: ${data.data.video_id}`);
  return { video_id: data.data.video_id };
}

async function main() {
  console.log('🔧 HEYGEN RESUBMIT MISSING VIDEOS');
  console.log('================================\n');
  console.log(`API Key: ${HEYGEN_API_KEY?.substring(0, 20)}...`);
  
  // Load existing manifest
  const manifestPath = 'generated-videos/day-351-manifest.json';
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  const results: Record<string, any> = {};
  
  for (const [archetype, avatarId] of Object.entries(MISSING_ARCHETYPES)) {
    try {
      const result = await submitVideo(archetype, avatarId);
      results[archetype] = {
        video_id: result.video_id,
        status: 'pending',
        submitted: new Date().toISOString(),
        resubmitted: true
      };
      
      // Update manifest
      manifest.videos[archetype] = {
        ...manifest.videos[archetype],
        video_id: result.video_id,
        status: 'pending',
        submitted: new Date().toISOString(),
        resubmitted: true
      };
      
    } catch (error: any) {
      console.log(`  ❌ Error: ${error.message}`);
      results[archetype] = { error: error.message };
    }
  }
  
  // Save updated manifest
  manifest.updated = new Date().toISOString();
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log('\n================================');
  console.log('RESULTS:');
  for (const [arch, result] of Object.entries(results)) {
    if (result.video_id) {
      console.log(`  ${arch}: ✅ ${result.video_id}`);
    } else {
      console.log(`  ${arch}: ❌ ${result.error}`);
    }
  }
  console.log(`\nManifest updated: ${manifestPath}`);
}

main().catch(console.error);
