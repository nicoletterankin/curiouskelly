/**
 * Generate ALL videos for a single day - ALL ages, ALL phases
 * Version 2: Better error handling and duplicate management
 */

const { neon } = require('@neondatabase/serverless');
require('dotenv').config();

const DATABASE_URL = 'postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require';
const sql = neon(DATABASE_URL);

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const HEYGEN_API_URL = 'https://api.heygen.com/v2/video/generate';
// HEYGEN voice ID - NOT ElevenLabs! The old ID 'wAdymQH5YucAkXwmrdL0' was ElevenLabs.
const KELLY_VOICE_ID = 'BbuMXx40WT4ZuAgRXvNx'; // HeyGen English female voice

// Custom Kelly talking photo IDs
const KELLY_AVATARS = {
  explorer: '45e5ef8b651846e0b62b7477e552e87b',
  scientist: '7bb18cddacd44333813cc90ffa44f766',
  storyteller: '9ffd06bd986a4e3086612921f3ac87ea',
  macgyver: 'b9032c922c6e4e35b58a98abd499d060',
  mystic: 'a2b31ed0b5f84b0fa02d15d411735d3a',
  rebel: 'e614671b193c40f99772f7de5d1c51f7'
};

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];
const AGES = ['kid', 'adult', 'senior'];

let stats = { submitted: 0, skipped: 0, errors: 0 };

async function getScript(day, phase) {
  // Try lesson_perspectives first
  const colMap = {
    hook: 'hook_script',
    story: 'story_script', 
    wonder: 'wonder_script',
    action: 'action_script',
    wisdom: 'wisdom_script'
  };
  
  try {
    const result = await sql`
      SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lesson_perspectives 
      WHERE day_number = ${day} AND language = 'en'
      LIMIT 1
    `;
    if (result[0]) {
      const script = result[0][colMap[phase]];
      if (script) return script;
    }
  } catch (e) {}
  
  // Try lesson_atoms
  try {
    const atoms = await sql`
      SELECT la.content->>'script' as script
      FROM lesson_atoms la
      JOIN core_lessons cl ON la.core_lesson_id = cl.id
      WHERE cl.day_number = ${day} AND la.phase = ${phase}
      LIMIT 1
    `;
    if (atoms[0]?.script) return atoms[0].script;
  } catch (e) {}
  
  // Get lesson topic for fallback
  const lesson = await sql`SELECT title, topic FROM core_lessons WHERE day_number = ${day} LIMIT 1`;
  const topic = lesson[0]?.topic || lesson[0]?.title || 'today\'s topic';
  
  const defaults = {
    hook: `Hey there! Today we're exploring ${topic}. Are you ready to discover something amazing?`,
    story: `Let me tell you about ${topic}. It's really fascinating!`,
    wonder: `Here's what's truly amazing about ${topic}...`,
    action: `Now let's explore ${topic} together! Here's something you can try.`,
    wisdom: `Remember what we learned about ${topic}. Keep being curious!`
  };
  
  return defaults[phase];
}

async function videoExistsAndComplete(day, phase, age, archetype) {
  const result = await sql`
    SELECT id, status, heygen_video_id, video_url
    FROM heygen_videos 
    WHERE day_of_year = ${day} AND phase = ${phase} 
      AND age_category = ${age} AND archetype = ${archetype}
    LIMIT 1
  `;
  
  if (result.length === 0) return { exists: false };
  
  const row = result[0];
  return {
    exists: true,
    id: row.id,
    status: row.status,
    hasVideoId: !!row.heygen_video_id,
    hasUrl: !!row.video_url
  };
}

async function submitToHeyGen(script, avatarId) {
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: avatarId
      },
      voice: {
        type: 'text',
        input_text: script,
        voice_id: KELLY_VOICE_ID
      }
    }],
    dimension: { width: 1280, height: 720 }
  };
  
  const response = await fetch(HEYGEN_API_URL, {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  const data = await response.json();
  
  if (data.error) {
    throw new Error(data.error.message || JSON.stringify(data.error));
  }
  if (!data.data?.video_id) {
    throw new Error('No video_id in response: ' + JSON.stringify(data));
  }
  
  return data.data.video_id;
}

async function generateOne(day, phase, age, archetype) {
  const key = `Day${day}/${age}/${archetype}/${phase}`;
  
  // Check existing status
  const existing = await videoExistsAndComplete(day, phase, age, archetype);
  
  if (existing.exists) {
    if (existing.status === 'completed' && existing.hasUrl) {
      console.log(`  ✅ ${key} - ALREADY COMPLETE`);
      stats.skipped++;
      return;
    }
    if (existing.status === 'processing' && existing.hasVideoId) {
      console.log(`  ⏳ ${key} - PROCESSING (${existing.status})`);
      stats.skipped++;
      return;
    }
  }
  
  // Get script
  const script = await getScript(day, phase);
  const avatarId = KELLY_AVATARS[archetype] || KELLY_AVATARS.explorer;
  
  try {
    // Submit to HeyGen
    const videoId = await submitToHeyGen(script, avatarId);
    console.log(`  📤 ${key} - SUBMITTED: ${videoId}`);
    
    // Save to database
    if (existing.exists) {
      await sql`
        UPDATE heygen_videos SET
          heygen_video_id = ${videoId},
          status = 'processing',
          script = ${script},
          updated_at = NOW()
        WHERE id = ${existing.id}
      `;
    } else {
      await sql`
        INSERT INTO heygen_videos (
          id, day_of_year, phase, age_category, archetype,
          heygen_video_id, status, script, language, created_at, updated_at
        ) VALUES (
          gen_random_uuid(), ${day}, ${phase}, ${age}, ${archetype},
          ${videoId}, 'processing', ${script}, 'en', NOW(), NOW()
        )
      `;
    }
    
    stats.submitted++;
  } catch (e) {
    console.log(`  ❌ ${key} - ERROR: ${e.message}`);
    stats.errors++;
  }
  
  // Rate limit - wait between requests
  await new Promise(r => setTimeout(r, 600));
}

async function generateDay(day) {
  console.log(`\n╔════════════════════════════════════════════════════════════╗`);
  console.log(`║  GENERATING DAY ${String(day).padStart(3)} - ALL AGES × ALL PHASES              ║`);
  console.log(`╚════════════════════════════════════════════════════════════╝\n`);
  
  // Map ages to archetypes for variety
  const ageArchetypes = {
    kid: 'explorer',
    adult: 'scientist', 
    senior: 'storyteller'
  };
  
  for (const age of AGES) {
    console.log(`\n🎭 ${age.toUpperCase()}:`);
    const archetype = ageArchetypes[age];
    
    for (const phase of PHASES) {
      await generateOne(day, phase, age, archetype);
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  const startDay = parseInt(args[0]) || 34;
  const endDay = parseInt(args[1]) || startDay;
  
  // Check credits
  const creditsRes = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const credits = await creditsRes.json();
  console.log(`💰 HeyGen Credits: ${credits.data?.remaining_quota || 'unknown'}`);
  
  for (let day = startDay; day <= endDay; day++) {
    await generateDay(day);
  }
  
  console.log(`\n════════════════════════════════════════════════════════════`);
  console.log(`📊 FINAL: ${stats.submitted} submitted, ${stats.skipped} skipped, ${stats.errors} errors`);
  console.log(`════════════════════════════════════════════════════════════`);
}

main().catch(console.error);
