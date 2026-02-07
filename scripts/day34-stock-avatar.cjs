#!/usr/bin/env node
/**
 * Generate Day 34 using STOCK HeyGen avatar
 * This is a quick fix to get SOMETHING playing today
 */
require('dotenv').config();
const fs = require('fs');
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// Use a stock female avatar that looks professional
const AVATAR_ID = 'Abigail_expressive_2024112501'; // Abigail - Upper Body
const VOICE_ID = 'BbuMXx40WT4ZuAgRXvNx'; // Kelly2 voice

const SCRIPTS = {
  hook: "What makes magnets attract and repel? Today we're exploring the invisible forces that shape our world.",
  story: "Long ago, ancient Greeks discovered lodestone, a rock that could move iron. They thought it was magic, but it was actually the first discovery of magnetism.",
  wonder: "What other invisible forces might be shaping our universe right now? Scientists are still discovering new things about magnetism.",
  action: "Try this: Take a magnet and test different objects. Make a list of what's magnetic and what isn't. You might be surprised!",
  wisdom: "The most powerful forces are often the ones we cannot see. Understanding them helps us predict and shape the future."
};

async function generateVideo(phase, script) {
  console.log(`Generating ${phase}...`);
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'avatar',
          avatar_id: AVATAR_ID,
          avatar_style: 'normal'
        },
        voice: {
          type: 'text',
          input_text: script,
          voice_id: VOICE_ID
        }
      }],
      dimension: { width: 1280, height: 720 }
    })
  });

  const data = await response.json();
  
  if (data.data?.video_id) {
    console.log(`  ✅ ${phase}: ${data.data.video_id}`);
    return { phase, videoId: data.data.video_id };
  } else {
    console.log(`  ❌ ${phase}: ${data.error?.message || JSON.stringify(data)}`);
    return { phase, error: data.error?.message };
  }
}

async function main() {
  console.log('DAY 34 - STOCK AVATAR (Abigail)');
  console.log('This is a temporary fix while we sort out account access');
  console.log('='.repeat(50));
  
  const results = [];
  
  for (const phase of ['hook', 'story', 'wonder', 'action', 'wisdom']) {
    const result = await generateVideo(phase, SCRIPTS[phase]);
    results.push(result);
    await new Promise(r => setTimeout(r, 2000));
  }
  
  const success = results.filter(r => r.videoId);
  
  console.log('');
  console.log('RESULTS:');
  success.forEach(r => console.log(`  ${r.phase}: ${r.videoId}`));
  
  if (success.length > 0) {
    fs.writeFileSync('day34-stock-videos.json', JSON.stringify({
      day: 34,
      timestamp: new Date().toISOString(),
      avatarId: AVATAR_ID,
      voiceId: VOICE_ID,
      videos: success
    }, null, 2));
    
    console.log('');
    console.log('Videos generating on HeyGen (2-5 minutes)');
    console.log('Then run: node scripts/sync-day34-to-db.cjs');
  }
  
  await pool.end();
}

main().catch(console.error);
