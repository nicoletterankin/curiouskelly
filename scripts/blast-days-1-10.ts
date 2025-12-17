#!/usr/bin/env npx tsx
/**
 * BLAST GENERATOR - Days 1-10
 * 
 * Uses ALL 6 API keys to generate high-quality 1:1 square visuals
 * with lesson-specific creative prompts (not generic garbage)
 */

import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as crypto from 'crypto';
import * as fs from 'fs';

dotenv.config({ path: path.join(process.cwd(), '.env.local') });

// ============================================================================
// CONFIGURATION
// ============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;

const supabase = createClient(SUPABASE_URL!, SUPABASE_KEY!);

// Collect ALL API keys
function getApiKeys(): string[] {
  const keys: string[] = [];
  const mainKey = process.env.GOOGLE_API_KEY;
  if (mainKey) keys.push(mainKey);
  
  for (let i = 1; i <= 10; i++) {
    const key = process.env[`GOOGLE_API_KEY_${i}`];
    if (key && !keys.includes(key)) keys.push(key);
  }
  return keys;
}

const API_KEYS = getApiKeys();
console.log(`🔑 Found ${API_KEYS.length} API keys`);

const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'] as const;
type Phase = typeof PHASES[number];

// Track key usage
const keyUsage: Map<string, number> = new Map();
const KEY_LIMIT = 30; // Ultra limit per key

// ============================================================================
// LESSON-SPECIFIC CREATIVE PROMPTS
// ============================================================================

const LESSON_PROMPTS: Record<number, Record<Phase, string>> = {
  1: { // Starting Fresh
    hook: `A single autumn leaf frozen mid-air between a tree branch and the ground, caught in the eternal moment of falling. Golden light, shallow depth of field. The pause before a new chapter begins.`,
    cliff: `A split image: left side shows a January 1st calendar with confetti, right side shows a random Tuesday morning with a door opening into pure light. Same energy, different day. Which matters more?`,
    q1: `A brain visualization with new neural pathways lighting up like city streets at night - glowing golden connections forming in real-time. Not a diagram - photorealistic, beautiful, alive.`,
    q2: `An hourglass where the sand is flowing UPWARD, defying gravity. Warm light. The past becoming the future. Time as possibility, not constraint.`,
    q3: `A hand writing in a journal, but the words are lifting off the page and becoming real objects - goals transforming into reality. Magical realism.`,
    wisdom: `A person standing at a threshold - a doorway made of morning light. Behind them, shadows. Ahead, golden possibility. The moment of choosing to begin.`,
    outro: `First footstep onto a new path, morning dew on grass, the trail ahead disappearing into beautiful mist. Forward momentum.`
  },
  
  2: { // Three Lives of Water
    hook: `A single water droplet that contains reflections of three scenes: a dinosaur drinking, a medieval well, a modern glass of water. Time compressed into one drop. Photorealistic, magical.`,
    cliff: `Side by side: a pot of boiling water and a tray of ice cubes. But the boiling water is frozen solid while ice remains. The Mpemba paradox visualized. Steam and frost.`,
    q1: `A water molecule shown as a beautiful geometric form, traveling through time - past glaciers, ancient oceans, dinosaur veins, human cells. The same molecule, endless journeys.`,
    q2: `Cross-section of Earth showing glowing blue hidden oceans deep in the mantle - three times more water than all surface oceans. The secret depths.`,
    q3: `Three versions of H2O - ice crystals, liquid flow, steam rising - but they're holding hands, the same molecule in different dances. Unity through transformation.`,
    wisdom: `A person standing in rain, arms open, face upward. The water droplets are glowing slightly - ancient travelers finally reaching their destination.`,
    outro: `A glass of water on a windowsill catching morning light, refracting rainbows. Simple. Profound. Billions of years in a glass.`
  },
  
  3: { // Where Clouds Come From
    hook: `A massive cumulus cloud on an old-fashioned scale, balanced against a pile of tiny, almost invisible water droplets. The impossible weight that floats.`,
    cliff: `An atmospheric river visualized - a glowing stream of water vapor in the sky, larger than the Mississippi River below. The invisible made visible.`,
    q1: `Extreme close-up of water droplets so small they're suspended by air molecules bouncing around them - like dust in a sunbeam but it's the secret of clouds.`,
    q2: `A time-lapse compressed into one image: a cloud forming, shifting shape, dissolving - three states in one frame. Impermanence made visible.`,
    q3: `100 different cloud types arranged like a taxonomy chart, but beautiful - cirrus wisps, thunderhead towers, fog banks - clouds as storytellers.`,
    wisdom: `A person looking up at clouds that are shaped like their worries - a deadline, a fear, a regret - but the clouds are already dissolving, drifting apart.`,
    outro: `Late afternoon sky, golden hour, clouds painted pink and orange. A reminder to look up.`
  },
  
  4: { // How Light Travels
    hook: `A person reaching toward the sun, but their hand is in sharp focus labeled "NOW" while the sun is slightly blurred, labeled "8 MINUTES AGO". The time gap we never notice.`,
    cliff: `The night sky where each star has a small timestamp floating near it - "2,000 years", "4 million years", "100 years" - we're seeing different eras simultaneously.`,
    q1: `Light racing around Earth's curve - 7.5 laps in one second. Motion blur, speed, the universal limit made visible.`,
    q2: `A telescope pointed at a star, but instead of seeing the star, we see what was happening on Earth when that light left - ancient civilizations, dinosaurs.`,
    q3: `A double exposure: light behaving as a wave overlaid with light as particles. The same phenomenon, two natures. Quantum strangeness.`,
    wisdom: `A human silhouette made entirely of soft, glowing light - warm golden inside, stardust made conscious. "You are light."`,
    outro: `Sunrise breaking over a horizon, the first rays reaching toward the viewer. Light that left 8 minutes ago, arriving now.`
  },
  
  5: { // How Sound Moves
    hook: `An explosion in deep space - fire, debris, chaos - but with a large "MUTE" symbol overlaid. Perfect silence. The violence no one can hear.`,
    cliff: `Sound waves racing through three mediums - air, water, steel - with steel's waves far ahead. Speed differences visualized.`,
    q1: `A human ear with a hydrogen atom shown for scale - impossibly tiny - yet our ears detect vibrations even smaller than this.`,
    q2: `An astronaut floating in the void, helmet off (artistic liberty), mouth open in a scream that produces nothing - no waves, no ripples, just silence.`,
    q3: `A single vibration traveling from a guitar string, becoming visible waves, entering an ear, transforming into meaning in a brain. The journey of sound.`,
    wisdom: `A person in a quiet room, eyes closed, with subtle visual waves emanating from their chest - the quiet sounds of their own heartbeat, breath, thoughts.`,
    outro: `A beautiful moment of two people in conversation, one truly listening - their attention visualized as soft light encompassing the speaker.`
  },
  
  6: { // What's Inside a Seed
    hook: `A date palm seed in the foreground, sharp focus. Behind it, ghostly: Roman soldiers, medieval castles, modern cities - 2,000 years of waiting in one frame.`,
    cliff: `A scale showing one million orchid seeds on one side, a single gram weight on the other - perfectly balanced. Impossibly light potential.`,
    q1: `A seed cracked open, but inside is a miniature tree glowing with biological light - the complete blueprint encoded, waiting to unfold.`,
    q2: `One sunflower head exploding into 2,000 smaller sunflowers, each exploding again - exponential abundance fractaling outward.`,
    q3: `Seeds entering suspended animation - their internal processes visualized as slowing, dimming, then holding steady. Patience encoded in DNA.`,
    wisdom: `A person shown as a seed that has sprouted - roots reaching down, growth reaching up, becoming what they're meant to be. Still growing.`,
    outro: `A child's hand holding seeds, dirt visible on the palm, ready to plant. Simple beginning of something vast.`
  },
  
  7: { // What Stars Are Made Of
    hook: `A drop of blood magnified, but inside the blood cells, we see tiny supernova explosions - the iron's origin story playing inside your veins.`,
    cliff: `A human figure made of stars and nebula gas, looking up at a constellation that's shaped like a human - the universe experiencing itself.`,
    q1: `A star's nuclear furnace visualized - hydrogen atoms smashing into helium, helium into carbon, building the periodic table through fire.`,
    q2: `A supernova explosion, but the expanding debris is labeled with elements - calcium, iron, gold - the guts of a dying star becoming your bones.`,
    q3: `A handful of beach sand next to a view of the galaxy - but the galaxy contains MORE points of light than grains in the hand. Incomprehensible scale.`,
    wisdom: `A newborn baby surrounded by soft starlight, as if the cosmos itself is cradling its creation - stardust that learned to breathe.`,
    outro: `Night sky, clear and vast, a person small below looking up - the witness and the witnessed, connected across light-years.`
  },
  
  8: { // What Makes a Real Friend
    hook: `A clock face showing 200 hours, with photos around the edge transforming: strangers at hour 0, casual friends at hour 50, close friends at hour 200.`,
    cliff: `A social network visualization - 500 faint gray connections, but only 2-3 glowing bright gold. The inner circle that actually matters.`,
    q1: `Two side-by-side health meters: one showing "Exercise + Diet" gains, one showing "Strong Friendships" gains - friendships adding MORE to the lifespan bar.`,
    q2: `A phone glowing at 3 AM in a dark room. On the screen: a contact name. Just one name. The person you'd actually call when everything falls apart.`,
    q3: `Two imperfect people - shown with visible flaws, cracks, rough edges - standing shoulder to shoulder. Imperfection accepted. Still together.`,
    wisdom: `Two old hands clasped together - weathered, spotted with age - decades of friendship visible in the comfortable grip.`,
    outro: `Two empty chairs facing each other in a garden, afternoon light, clearly having just held a long conversation. Presence even in absence.`
  },
  
  9: { // How Kindness Spreads
    hook: `A stone dropping into water, but the ripples are made of tiny human figures - each ring showing people affected, spreading outward to strangers.`,
    cliff: `A chain reaction visualized: Person 1 holds a door → Person 2 smiles → Person 3 tips generously → Person 4 helps a stranger. Invisible connections.`,
    q1: `A brain scan visualization showing dopamine, serotonin, and oxytocin lighting up like fireworks - the chemical reward of giving.`,
    q2: `Three degrees of separation visualized: You → Your friend → Their friend → A stranger you'll never meet. Your kindness reaching the unknown.`,
    q3: `The "elevation effect" - a person witnessing kindness between strangers, and their own heart beginning to glow with warmth. Contagious goodness.`,
    wisdom: `A single kind act - a small help, almost invisible - but shown as a seed that's already grown into a forest behind the giver. Unseen impact.`,
    outro: `An anonymous hand leaving something kind - a note, a paid coffee, a small gift - walking away without looking back. Pure giving.`
  },
  
  10: { // The Art of Really Listening
    hook: `A conversation where 75% of the words are fading, transparent, dissolving - only 25% remain solid. What we actually retain when we don't really listen.`,
    cliff: `A person "listening" but their head is crowded with floating distractions: phone, to-do list, their own "brilliant comeback" thought bubble. The noise of not listening.`,
    q1: `A split-screen brain: one side processing at 125 words/minute (the speaker), one side capable of 400 words/minute - the gap where attention wanders.`,
    q2: `Two people in conversation, but the listener's full attention is visualized as a warm beam of light wrapping around the speaker - attention as a gift.`,
    q3: `A speaker's stress hormones (cortisol) visibly decreasing as they're truly heard - the physiological power of being listened to.`,
    wisdom: `Two people, one speaking, one listening with complete presence - no thought bubbles, no distractions, just pure attention. Rare and beautiful.`,
    outro: `An ear transforming into an open doorway - listening as an invitation inward. The gift of being understood.`
  }
};

// ============================================================================
// GENERATION
// ============================================================================

async function generateImage(prompt: string, apiKey: string): Promise<Buffer | null> {
  const fullPrompt = `${prompt}

STYLE: Ultra photorealistic, cinematic quality. 1:1 square aspect ratio. Dramatic lighting. Emotional resonance. NOT a diagram. NOT clipart. Professional photography aesthetic.

DO NOT include any text, labels, words, or writing in the image.`;

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-ultra-generate-001:predict?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instances: [{ prompt: fullPrompt }],
          parameters: {
            sampleCount: 1,
            aspectRatio: '1:1',
            safetySetting: 'block_low_and_above'
          }
        })
      }
    );

    if (!response.ok) {
      const err = await response.json();
      if (response.status === 429 || err.error?.status === 'RESOURCE_EXHAUSTED') {
        keyUsage.set(apiKey, KEY_LIMIT); // Mark exhausted
        return null;
      }
      console.log(`      ⚠️ Error: ${err.error?.message?.substring(0, 50)}`);
      return null;
    }

    const result = await response.json();
    const imageData = result.predictions?.[0]?.bytesBase64Encoded;
    if (!imageData) return null;

    keyUsage.set(apiKey, (keyUsage.get(apiKey) || 0) + 1);
    return Buffer.from(imageData, 'base64');
  } catch (error: any) {
    console.log(`      ⚠️ ${error.message?.substring(0, 50)}`);
    return null;
  }
}

function getAvailableKey(): string | null {
  for (const key of API_KEYS) {
    if ((keyUsage.get(key) || 0) < KEY_LIMIT) {
      return key;
    }
  }
  return null;
}

async function saveVisual(
  dayNumber: number,
  phase: Phase,
  topic: string,
  buffer: Buffer,
  prompt: string
): Promise<boolean> {
  const hash = crypto.createHash('sha256')
    .update(JSON.stringify({ d: dayNumber, p: phase, v: 'blast-v1' }))
    .digest('hex');
  
  const storagePath = `blast/${dayNumber}/${phase}.png`;
  
  const { error: uploadError } = await supabase.storage
    .from('visuals')
    .upload(storagePath, buffer, { contentType: 'image/png', upsert: true });
    
  if (uploadError) {
    console.log(`      ❌ Upload: ${uploadError.message}`);
    return false;
  }
  
  const { data: urlData } = supabase.storage.from('visuals').getPublicUrl(storagePath);
  
  const { error: dbError } = await supabase
    .from('visual_commons')
    .upsert({
      content_hash: hash,
      day_number: dayNumber,
      phase,
      topic,
      visual_type: 'scene',
      age_group: 'all',
      style: 'blast-v1',
      storage_path: storagePath,
      public_url: urlData.publicUrl,
      format: 'png',
      prompt_used: prompt,
      model_used: 'imagen-4.0-ultra-generate-001',
      generation_params: { aspectRatio: '1:1', version: 'blast-v1' },
      estimated_cost: 0.06,
      generated_by_display_name: 'Blast Generator',
      generation_source: 'blast',
      status: 'active'
    }, { onConflict: 'content_hash' });
    
  return !dbError;
}

async function checkExists(dayNumber: number, phase: Phase): Promise<boolean> {
  const hash = crypto.createHash('sha256')
    .update(JSON.stringify({ d: dayNumber, p: phase, v: 'blast-v1' }))
    .digest('hex');
    
  const { data } = await supabase
    .from('visual_commons')
    .select('id')
    .eq('content_hash', hash)
    .single();
    
  return !!data;
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  console.log('═'.repeat(60));
  console.log('🚀 BLAST GENERATOR - Days 1-10');
  console.log('═'.repeat(60));
  console.log(`🔑 Keys: ${API_KEYS.length}`);
  console.log(`📊 Max capacity: ${API_KEYS.length * KEY_LIMIT} images`);
  console.log(`🎯 Generating: 10 days × 7 phases = 70 images`);
  
  // Load lesson topics
  const topics: Record<number, string> = {};
  for (let d = 1; d <= 10; d++) {
    const lessonPath = path.join(process.cwd(), 'public', 'lessons', `day-${d}.json`);
    if (fs.existsSync(lessonPath)) {
      const lesson = JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
      topics[d] = lesson.meta.topic;
    }
  }
  
  let generated = 0;
  let skipped = 0;
  let failed = 0;
  
  for (let day = 1; day <= 10; day++) {
    console.log(`\n📚 Day ${day}: ${topics[day] || 'Unknown'}`);
    
    for (const phase of PHASES) {
      // Check if exists
      if (await checkExists(day, phase)) {
        console.log(`   ⏭️ ${phase}: exists`);
        skipped++;
        continue;
      }
      
      const prompt = LESSON_PROMPTS[day]?.[phase];
      if (!prompt) {
        console.log(`   ⚠️ ${phase}: no prompt`);
        failed++;
        continue;
      }
      
      const apiKey = getAvailableKey();
      if (!apiKey) {
        console.log(`   ❌ ${phase}: all keys exhausted`);
        failed++;
        continue;
      }
      
      const keyIndex = API_KEYS.indexOf(apiKey) + 1;
      process.stdout.write(`   🎨 ${phase}: Key${keyIndex}...`);
      
      const buffer = await generateImage(prompt, apiKey);
      if (!buffer) {
        console.log(' ❌ failed');
        failed++;
        continue;
      }
      
      const saved = await saveVisual(day, phase, topics[day] || '', buffer, prompt);
      if (saved) {
        console.log(' ✅');
        generated++;
      } else {
        console.log(' ❌ save failed');
        failed++;
      }
      
      // Small delay to be nice
      await new Promise(r => setTimeout(r, 500));
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 RESULTS');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${generated}`);
  console.log(`⏭️ Skipped: ${skipped}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`💰 Cost: $${(generated * 0.06).toFixed(2)}`);
  
  // Show key usage
  console.log('\n🔑 Key Usage:');
  API_KEYS.forEach((key, i) => {
    console.log(`   Key ${i + 1}: ${keyUsage.get(key) || 0}/${KEY_LIMIT}`);
  });
}

main().catch(console.error);
