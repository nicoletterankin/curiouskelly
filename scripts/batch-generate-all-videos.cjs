/**
 * INDUSTRIAL-SCALE VIDEO GENERATION
 * 
 * This script generates ALL videos systematically:
 * - All 365 days
 * - All 5 phases (hook, story, wonder, action, wisdom)
 * - All 3 age categories (kid, adult, senior)
 * 
 * Uses custom Kelly avatars, proper rate limiting, and database tracking.
 */

const { neon } = require('@neondatabase/serverless');
require('dotenv').config();

// Database connection
const DATABASE_URL = 'postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require';
const sql = neon(DATABASE_URL);

// HeyGen configuration
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY || 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const HEYGEN_API_URL = 'https://api.heygen.com/v2/video/generate';

// Kelly's voice
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

// Custom Kelly avatars (Talking Photos) - NOT public avatars!
const KELLY_AVATARS = {
  kid: {
    explorer: '45e5ef8b651846e0b62b7477e552e87b',
    scientist: '7bb18cddacd44333813cc90ffa44f766',
    storyteller: '9ffd06bd986a4e3086612921f3ac87ea',
    macgyver: 'b9032c922c6e4e35b58a98abd499d060',
    mystic: 'a2b31ed0b5f84b0fa02d15d411735d3a',
    rebel: 'e614671b193c40f99772f7de5d1c51f7'
  },
  adult: {
    explorer: '45e5ef8b651846e0b62b7477e552e87b',
    scientist: '7bb18cddacd44333813cc90ffa44f766',
    storyteller: '9ffd06bd986a4e3086612921f3ac87ea',
    macgyver: 'b9032c922c6e4e35b58a98abd499d060',
    mystic: 'a2b31ed0b5f84b0fa02d15d411735d3a',
    rebel: 'e614671b193c40f99772f7de5d1c51f7'
  },
  senior: {
    explorer: '45e5ef8b651846e0b62b7477e552e87b',
    scientist: '7bb18cddacd44333813cc90ffa44f766',
    storyteller: '9ffd06bd986a4e3086612921f3ac87ea',
    macgyver: 'b9032c922c6e4e35b58a98abd499d060',
    mystic: 'a2b31ed0b5f84b0fa02d15d411735d3a',
    rebel: 'e614671b193c40f99772f7de5d1c51f7'
  }
};

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];
const AGE_CATEGORIES = ['kid', 'adult', 'senior'];
const ARCHETYPES = ['explorer', 'scientist', 'storyteller', 'macgyver', 'mystic', 'rebel'];

// Rate limiting: HeyGen allows ~10 concurrent, 100/min
const CONCURRENT_LIMIT = 5;
const DELAY_BETWEEN_BATCHES_MS = 3000;

// Stats tracking
let stats = {
  submitted: 0,
  skipped: 0,
  errors: 0,
  creditsUsed: 0
};

async function checkCredits() {
  const response = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const data = await response.json();
  return data.data?.remaining_quota || 0;
}

async function getScriptForVideo(day, phase, age, archetype) {
  // Try to get from lesson_perspectives first
  const perspectives = await sql`
    SELECT ${phase}_script as script 
    FROM lesson_perspectives 
    WHERE day_number = ${day} 
      AND age_group = ${age}
      AND archetype = ${archetype}
      AND language = 'en'
    LIMIT 1
  `;
  
  if (perspectives[0]?.script) {
    return perspectives[0].script;
  }
  
  // Fall back to lesson_atoms
  const atoms = await sql`
    SELECT la.content->>'script' as script
    FROM lesson_atoms la
    JOIN core_lessons cl ON la.core_lesson_id = cl.id
    WHERE cl.day_number = ${day}
      AND la.phase = ${phase}
      AND la.archetype = ${archetype}
    LIMIT 1
  `;
  
  if (atoms[0]?.script) {
    return atoms[0].script;
  }
  
  // Final fallback - generic script
  const lessons = await sql`
    SELECT title, topic FROM core_lessons WHERE day_number = ${day} LIMIT 1
  `;
  
  if (lessons[0]) {
    return `Today we're exploring ${lessons[0].topic || lessons[0].title}. Let's discover something wonderful together!`;
  }
  
  return null;
}

async function videoExists(day, phase, age, archetype) {
  const existing = await sql`
    SELECT id FROM heygen_videos 
    WHERE day_of_year = ${day} 
      AND phase = ${phase} 
      AND age_category = ${age} 
      AND archetype = ${archetype}
      AND status IN ('completed', 'processing')
    LIMIT 1
  `;
  return existing.length > 0;
}

async function submitVideo(day, phase, age, archetype, script) {
  const avatarId = KELLY_AVATARS[age]?.[archetype] || KELLY_AVATARS.adult.explorer;
  
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
    dimension: { width: 1280, height: 720 },
    aspect_ratio: '16:9'
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
    throw new Error(data.error?.message || 'Failed to submit video');
  }
  
  return data.data.video_id;
}

async function insertVideoRecord(day, phase, age, archetype, heygenVideoId, script) {
  await sql`
    INSERT INTO heygen_videos (
      id, day_of_year, phase, age_category, archetype, 
      heygen_video_id, status, script, language, created_at, updated_at
    ) VALUES (
      gen_random_uuid(), ${day}, ${phase}, ${age}, ${archetype},
      ${heygenVideoId}, 'processing', ${script}, 'en', NOW(), NOW()
    )
    ON CONFLICT (day_of_year, phase, age_category, archetype) 
    DO UPDATE SET 
      heygen_video_id = ${heygenVideoId},
      status = 'processing',
      script = ${script},
      updated_at = NOW()
  `;
}

async function generateVideo(day, phase, age, archetype) {
  // Check if already exists
  if (await videoExists(day, phase, age, archetype)) {
    stats.skipped++;
    return { status: 'skipped', reason: 'exists' };
  }
  
  // Get script
  const script = await getScriptForVideo(day, phase, age, archetype);
  if (!script) {
    stats.skipped++;
    return { status: 'skipped', reason: 'no script' };
  }
  
  try {
    // Submit to HeyGen
    const videoId = await submitVideo(day, phase, age, archetype, script);
    
    // Record in database
    await insertVideoRecord(day, phase, age, archetype, videoId, script);
    
    stats.submitted++;
    stats.creditsUsed += 1.5; // Estimate
    
    return { status: 'submitted', videoId };
  } catch (error) {
    stats.errors++;
    return { status: 'error', error: error.message };
  }
}

async function generateDay(day) {
  console.log(`\n📅 Generating Day ${day}...`);
  
  const tasks = [];
  for (const age of AGE_CATEGORIES) {
    for (const phase of PHASES) {
      // Use one archetype per age for efficiency (can expand later)
      const archetype = ARCHETYPES[AGE_CATEGORIES.indexOf(age) % ARCHETYPES.length];
      tasks.push({ day, phase, age, archetype });
    }
  }
  
  // Process in batches
  for (let i = 0; i < tasks.length; i += CONCURRENT_LIMIT) {
    const batch = tasks.slice(i, i + CONCURRENT_LIMIT);
    const results = await Promise.all(
      batch.map(t => generateVideo(t.day, t.phase, t.age, t.archetype))
    );
    
    results.forEach((r, idx) => {
      const t = batch[idx];
      if (r.status === 'submitted') {
        console.log(`  ✓ ${t.age}/${t.phase} submitted`);
      } else if (r.status === 'skipped') {
        console.log(`  - ${t.age}/${t.phase} skipped (${r.reason})`);
      } else {
        console.log(`  ✗ ${t.age}/${t.phase} error: ${r.error}`);
      }
    });
    
    if (i + CONCURRENT_LIMIT < tasks.length) {
      await new Promise(r => setTimeout(r, DELAY_BETWEEN_BATCHES_MS));
    }
  }
}

async function syncCompletedVideos() {
  console.log('\n🔄 Syncing completed videos...');
  
  const processing = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category
    FROM heygen_videos 
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    LIMIT 50
  `;
  
  let synced = 0;
  for (const video of processing) {
    try {
      const response = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${video.heygen_video_id}`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY }
      });
      const data = await response.json();
      
      if (data.data?.status === 'completed' && data.data?.video_url) {
        await sql`
          UPDATE heygen_videos 
          SET status = 'completed', 
              video_url = ${data.data.video_url},
              updated_at = NOW()
          WHERE id = ${video.id}
        `;
        synced++;
        console.log(`  ✓ Day ${video.day_of_year} ${video.phase} (${video.age_category}) - COMPLETED`);
      } else if (data.data?.status === 'failed') {
        await sql`
          UPDATE heygen_videos 
          SET status = 'failed', 
              error_message = ${data.data.error || 'Unknown error'},
              updated_at = NOW()
          WHERE id = ${video.id}
        `;
        console.log(`  ✗ Day ${video.day_of_year} ${video.phase} (${video.age_category}) - FAILED`);
      }
    } catch (e) {
      console.log(`  ? Error checking ${video.heygen_video_id}: ${e.message}`);
    }
    
    // Rate limit
    await new Promise(r => setTimeout(r, 200));
  }
  
  console.log(`  Synced ${synced} videos`);
  return synced;
}

async function main() {
  const args = process.argv.slice(2);
  const startDay = parseInt(args[0]) || 1;
  const endDay = parseInt(args[1]) || 365;
  const syncOnly = args.includes('--sync');
  
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║       INDUSTRIAL-SCALE HEYGEN VIDEO GENERATION             ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  // Check credits first
  const credits = await checkCredits();
  console.log(`\n💰 Available credits: ${credits}`);
  
  if (credits < 10) {
    console.log('❌ Not enough credits! Need at least 10 to continue.');
    process.exit(1);
  }
  
  if (syncOnly) {
    await syncCompletedVideos();
    process.exit(0);
  }
  
  // Calculate what we can generate with available credits
  const videosPerDay = AGE_CATEGORIES.length * PHASES.length; // 15 videos/day
  const creditsPerDay = videosPerDay * 1.5; // ~22.5 credits/day
  const maxDays = Math.floor(credits / creditsPerDay);
  
  console.log(`\n📊 Generation Plan:`);
  console.log(`   Days requested: ${startDay} to ${endDay} (${endDay - startDay + 1} days)`);
  console.log(`   Videos per day: ${videosPerDay}`);
  console.log(`   Credits per day: ~${creditsPerDay}`);
  console.log(`   Max days with current credits: ${maxDays}`);
  
  const actualEndDay = Math.min(endDay, startDay + maxDays - 1);
  console.log(`   Actual generation: Day ${startDay} to Day ${actualEndDay}`);
  
  // Confirm
  console.log(`\n⚡ Starting generation in 5 seconds... (Ctrl+C to cancel)`);
  await new Promise(r => setTimeout(r, 5000));
  
  // Generate
  for (let day = startDay; day <= actualEndDay; day++) {
    await generateDay(day);
    
    // Check credits periodically
    if (day % 5 === 0) {
      const remaining = await checkCredits();
      console.log(`\n💰 Credits remaining: ${remaining}`);
      if (remaining < 15) {
        console.log('⚠️ Low credits, stopping generation');
        break;
      }
    }
  }
  
  // Final sync
  await syncCompletedVideos();
  
  // Summary
  console.log('\n════════════════════════════════════════════════════════════');
  console.log('📊 GENERATION SUMMARY:');
  console.log(`   Submitted: ${stats.submitted}`);
  console.log(`   Skipped: ${stats.skipped}`);
  console.log(`   Errors: ${stats.errors}`);
  console.log(`   Est. credits used: ~${Math.round(stats.creditsUsed)}`);
  console.log('════════════════════════════════════════════════════════════');
}

main().catch(console.error);
