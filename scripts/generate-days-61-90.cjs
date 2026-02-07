require('dotenv').config();
const {Pool} = require('pg');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

const STORYTELLER_AVATAR = '9ffd06bd986a4e3086612921f3ac87ea';
const KELLY_VOICE = 'BbuMXx40WT4ZuAgRXvNx';
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

// Topics for Days 61-90
const TOPICS = {
  61: 'How Volcanoes Erupt', 62: 'Why Stars Twinkle', 63: 'How Bees Make Honey',
  64: 'Why Ice Floats', 65: 'How Earthquakes Happen', 66: 'Why Leaves Change Color',
  67: 'How Sound Travels', 68: 'Why We Get Hiccups', 69: 'How Clouds Form',
  70: 'Why the Moon Changes Shape', 71: 'How Plants Drink Water', 72: 'Why Fire is Hot',
  73: 'How Airplanes Fly', 74: 'Why We See Lightning Before Thunder', 75: 'How Spiders Make Webs',
  76: 'Why Soap Makes Bubbles', 77: 'How Our Eyes See Color', 78: 'Why Boats Float',
  79: 'How Batteries Work', 80: 'Why We Get Dizzy', 81: 'How Glaciers Move',
  82: 'Why Popcorn Pops', 83: 'How Bridges Stay Up', 84: 'Why We Yawn',
  85: 'How Fossils Form', 86: 'Why Mirrors Reflect', 87: 'How Muscles Work',
  88: 'Why Onions Make Us Cry', 89: 'How Tides Work', 90: 'Why We Have Fingerprints'
};

const SCRIPTS = {
  hook: (topic) => `Have you ever wondered ${topic.toLowerCase().replace('how ', 'how ').replace('why ', 'why ')}? Today we're going to explore this fascinating question together!`,
  story: (topic) => `Let me tell you an amazing story about ${topic.toLowerCase()}. Scientists have been curious about this for centuries, and what they discovered will blow your mind!`,
  wonder: (topic) => `Here's what makes ${topic.toLowerCase()} so incredible - there's actually a beautiful scientific explanation that connects to so many other things in our world!`,
  action: (topic) => `Now it's your turn! Here's a simple experiment you can try at home to see ${topic.toLowerCase()} in action. You'll need just a few everyday items.`,
  wisdom: (topic) => `What we learned about ${topic.toLowerCase()} teaches us something profound - that curiosity is the key to understanding our amazing universe. Keep asking questions!`
};

let generated = 0;
let errors = 0;

async function generateVideo(day, phase) {
  const topic = TOPICS[day];
  const script = SCRIPTS[phase](topic);
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: { 'X-Api-Key': HEYGEN_API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_inputs: [{
        character: { type: 'talking_photo', talking_photo_id: STORYTELLER_AVATAR },
        voice: { type: 'text', input_text: script, voice_id: KELLY_VOICE }
      }],
      dimension: { width: 1920, height: 1080 }
    })
  });
  
  const data = await response.json();
  if (data.error) {
    console.log(`❌ Day ${day} ${phase}: ${data.error.message || 'Error'}`);
    errors++;
    return null;
  }
  
  const videoId = data.data?.video_id;
  
  await pool.query(`
    INSERT INTO heygen_videos (id, day_of_year, phase, age_category, archetype, heygen_video_id, status, script, created_at, updated_at)
    VALUES (gen_random_uuid(), $1, $2, 'adult', 'storyteller', $3, 'processing', $4, NOW(), NOW())
    ON CONFLICT (day_of_year, phase, age_category, archetype) 
    DO UPDATE SET heygen_video_id = $3, status = 'processing', script = $4, updated_at = NOW()
  `, [day, phase, videoId, script]).catch(() => {});
  
  generated++;
  console.log(`✅ Day ${day} ${phase} (${generated}/150)`);
  return videoId;
}

async function main() {
  console.log('=== GENERATING DAYS 61-90 (150 videos) ===\n');
  
  for (let day = 61; day <= 90; day++) {
    console.log(`\n--- Day ${day}: ${TOPICS[day]} ---`);
    for (const phase of PHASES) {
      await generateVideo(day, phase);
      await new Promise(r => setTimeout(r, 2000)); // Rate limit
    }
  }
  
  console.log(`\n=== COMPLETE: ${generated} generated, ${errors} errors ===`);
  await pool.end();
}

main().catch(console.error);
