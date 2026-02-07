require('dotenv').config();
const {Pool} = require('pg');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

// REAL Curious Kelly avatar IDs
const ADULT_AVATARS = {
  storyteller: '9ffd06bd986a4e3086612921f3ac87ea',
  scientist: '7bb18cddacd44333813cc90ffa44f766',
  explorer: '45e5ef8b651846e0b62b7477e552e87b',
  rebel: 'e614671b193c40f99772f7de5d1c51f7',
  macgyver: 'b9032c922c6e4e35b58a98abd499d060'
};

const KELLY_VOICE = 'BbuMXx40WT4ZuAgRXvNx';

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

const TOPICS = {
  35: 'How Rainbows Form',
  36: 'Why We Dream',
  37: 'How Computers Think',
  38: 'Why the Ocean is Salty',
  39: 'How Birds Fly',
  40: 'Why We Have Seasons'
};

const SCRIPTS = {
  hook: (topic) => `Have you ever wondered ${topic.toLowerCase().replace('how ', 'how ').replace('why ', 'why ')}? Today we're going to explore this fascinating question together!`,
  story: (topic) => `Let me tell you an amazing story about ${topic.toLowerCase()}. Scientists have been curious about this for centuries, and what they discovered will blow your mind!`,
  wonder: (topic) => `Here's what makes ${topic.toLowerCase()} so incredible - there's actually a beautiful scientific explanation that connects to so many other things in our world!`,
  action: (topic) => `Now it's your turn! Here's a simple experiment you can try at home to see ${topic.toLowerCase()} in action. You'll need just a few everyday items.`,
  wisdom: (topic) => `What we learned about ${topic.toLowerCase()} teaches us something profound - that curiosity is the key to understanding our amazing universe. Keep asking questions!`
};

async function generateVideo(day, phase, archetype) {
  const topic = TOPICS[day];
  const script = SCRIPTS[phase](topic);
  const avatarId = ADULT_AVATARS[archetype];
  
  console.log(`Generating Day ${day} ${phase} (${archetype})...`);
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: avatarId
        },
        voice: {
          type: 'text',
          input_text: script,
          voice_id: KELLY_VOICE
        }
      }],
      dimension: { width: 1920, height: 1080 }
    })
  });
  
  const data = await response.json();
  
  if (data.error) {
    console.log(`  ERROR: ${data.error.message || JSON.stringify(data.error)}`);
    return null;
  }
  
  const videoId = data.data?.video_id;
  console.log(`  Video ID: ${videoId}`);
  
  // Save to database - use ON CONFLICT
  await pool.query(`
    INSERT INTO heygen_videos (id, day_of_year, phase, age_category, archetype, heygen_video_id, status, script, created_at, updated_at)
    VALUES (gen_random_uuid(), $1, $2, 'adult', $3, $4, 'processing', $5, NOW(), NOW())
    ON CONFLICT (day_of_year, phase, age_category, archetype) 
    DO UPDATE SET heygen_video_id = $4, status = 'processing', script = $5, updated_at = NOW()
  `, [day, phase, archetype, videoId, script]).catch(async (e) => {
    // If unique constraint on heygen_video_id fails, just update
    console.log(`  DB: Updating existing record`);
    await pool.query(`
      UPDATE heygen_videos SET status = 'processing', updated_at = NOW()
      WHERE heygen_video_id = $1
    `, [videoId]);
  });
  
  return videoId;
}

async function main() {
  console.log('=== GENERATING DAYS 35-40 ===\n');
  console.log('Credits: 616 minutes remaining\n');
  
  const videoIds = [];
  
  for (const day of [35, 36, 37, 38, 39, 40]) {
    console.log(`\n--- DAY ${day}: ${TOPICS[day]} ---`);
    for (const phase of PHASES) {
      const archetype = 'storyteller'; // Use storyteller for all phases
      const videoId = await generateVideo(day, phase, archetype);
      if (videoId) videoIds.push({ day, phase, archetype, videoId });
      
      // Rate limit: wait 2 seconds between requests
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  
  console.log(`\n=== SUBMITTED ${videoIds.length} VIDEOS ===`);
  console.log('Video IDs saved to database with status "processing"');
  console.log('Run sync script in 5 minutes to fetch completed URLs');
  
  // Save video IDs to file for syncing
  require('fs').writeFileSync(
    'heygen-days-35-40-progress.json',
    JSON.stringify({ videoIds, startedAt: new Date().toISOString() }, null, 2)
  );
  
  await pool.end();
}

main().catch(console.error);
