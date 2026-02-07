/**
 * Generate ALL videos for a single day - ALL ages, ALL phases, ALL archetypes
 * 
 * Usage: node scripts/generate-day-full.cjs 34
 */

const { neon } = require('@neondatabase/serverless');
require('dotenv').config();

const DATABASE_URL = 'postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require';
const sql = neon(DATABASE_URL);

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const HEYGEN_API_URL = 'https://api.heygen.com/v2/video/generate';
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

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

// Default scripts per phase if not found in DB
const DEFAULT_SCRIPTS = {
  hook: "Hey there! Something amazing is about to happen. Are you ready to discover something incredible today?",
  story: "Let me tell you a story that will change how you see the world. It all started when...",
  wonder: "Now here's where it gets really interesting. Have you ever wondered why...?",
  action: "Your turn! Here's something fun you can try right now to explore this yourself.",
  wisdom: "Remember, every day is a chance to learn something new. Keep being curious!"
};

let stats = { submitted: 0, skipped: 0, errors: 0 };

async function getScript(day, phase, age, archetype) {
  // Try lesson_perspectives
  const phaseCol = `${phase}_script`;
  try {
    const result = await sql`
      SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
      FROM lesson_perspectives 
      WHERE day_number = ${day} AND language = 'en'
      LIMIT 1
    `;
    if (result[0] && result[0][phaseCol]) {
      return result[0][phaseCol];
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
    if (atoms[0]?.script) {
      return atoms[0].script;
    }
  } catch (e) {}
  
  // Get lesson topic for default
  try {
    const lesson = await sql`SELECT title, topic FROM core_lessons WHERE day_number = ${day} LIMIT 1`;
    if (lesson[0]) {
      const topic = lesson[0].topic || lesson[0].title;
      if (phase === 'hook') return `Hey there! Today we're going to explore ${topic}. Ready?`;
      if (phase === 'story') return `Let me tell you about ${topic}. It's really fascinating!`;
      if (phase === 'wonder') return `Here's what's amazing about ${topic}...`;
      if (phase === 'action') return `Now let's explore ${topic} together!`;
      if (phase === 'wisdom') return `Remember what we learned about ${topic} today!`;
    }
  } catch (e) {}
  
  return DEFAULT_SCRIPTS[phase];
}

async function videoExists(day, phase, age, archetype) {
  const result = await sql`
    SELECT id FROM heygen_videos 
    WHERE day_of_year = ${day} AND phase = ${phase} 
      AND age_category = ${age} AND archetype = ${archetype}
      AND status IN ('completed', 'processing')
  `;
  return result.length > 0;
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
  if (data.error || !data.data?.video_id) {
    throw new Error(data.error?.message || JSON.stringify(data));
  }
  return data.data.video_id;
}

async function insertRecord(day, phase, age, archetype, videoId, script) {
  // Check if exists first
  const existing = await sql`
    SELECT id FROM heygen_videos 
    WHERE day_of_year = ${day} AND phase = ${phase} 
      AND age_category = ${age} AND archetype = ${archetype}
  `;
  
  if (existing.length > 0) {
    // Update existing
    await sql`
      UPDATE heygen_videos SET
        heygen_video_id = ${videoId},
        status = 'processing',
        script = ${script},
        updated_at = NOW()
      WHERE day_of_year = ${day} AND phase = ${phase} 
        AND age_category = ${age} AND archetype = ${archetype}
    `;
  } else {
    // Insert new
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
}

async function generateOne(day, phase, age, archetype) {
  const key = `Day${day}/${age}/${archetype}/${phase}`;
  
  if (await videoExists(day, phase, age, archetype)) {
    console.log(`  ⏭️  ${key} - EXISTS`);
    stats.skipped++;
    return;
  }
  
  const script = await getScript(day, phase, age, archetype);
  const avatarId = KELLY_AVATARS[archetype] || KELLY_AVATARS.explorer;
  
  try {
    const videoId = await submitToHeyGen(script, avatarId);
    await insertRecord(day, phase, age, archetype, videoId, script);
    console.log(`  ✅ ${key} - SUBMITTED (${videoId})`);
    stats.submitted++;
  } catch (e) {
    console.log(`  ❌ ${key} - ERROR: ${e.message}`);
    stats.errors++;
  }
  
  // Rate limit
  await new Promise(r => setTimeout(r, 500));
}

async function generateDay(day) {
  console.log(`\n╔════════════════════════════════════════════════════════════╗`);
  console.log(`║  GENERATING DAY ${day} - ALL AGES, ALL PHASES                 ║`);
  console.log(`╚════════════════════════════════════════════════════════════╝\n`);
  
  // For each age, use a different primary archetype
  const ageArchetypes = {
    kid: 'explorer',
    adult: 'scientist', 
    senior: 'storyteller'
  };
  
  for (const age of AGES) {
    console.log(`\n🎭 ${age.toUpperCase()} AGE CATEGORY:`);
    const archetype = ageArchetypes[age];
    
    for (const phase of PHASES) {
      await generateOne(day, phase, age, archetype);
    }
  }
}

async function main() {
  const day = parseInt(process.argv[2]) || 34;
  
  // Check credits first
  const creditsRes = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const credits = await creditsRes.json();
  console.log(`💰 Credits available: ${credits.data?.remaining_quota || 'unknown'}`);
  
  await generateDay(day);
  
  console.log(`\n════════════════════════════════════════════════════════════`);
  console.log(`📊 RESULTS: ${stats.submitted} submitted, ${stats.skipped} skipped, ${stats.errors} errors`);
  console.log(`════════════════════════════════════════════════════════════`);
}

main().catch(console.error);
