#!/usr/bin/env node
/**
 * URGENT: Generate videos for TODAY (Day 34)
 * This ensures visitors see lip-synced Kelly
 */
require('dotenv').config();
const fs = require('fs');
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const TODAY = 34; // Day 34 of 2026
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

// Adult storyteller avatar (most common visitor)
const AVATAR_ID = '9ffd06bd986a4e3086612921f3ac87ea';
const VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'; // ElevenLabs Kelly voice

async function getScript(day, phase) {
  const result = await pool.query(`
    SELECT ${phase}_script as script 
    FROM lesson_perspectives 
    WHERE day_number = $1 AND age_group = 'adult' AND archetype = 'storyteller'
    LIMIT 1
  `, [day]);
  
  if (result.rows[0]?.script) {
    return result.rows[0].script;
  }
  
  // Fallback to lessons table
  const fallback = await pool.query(`
    SELECT hook_script, story_script, wonder_script, action_script, wisdom_script
    FROM lessons WHERE day_number = $1
  `, [day]);
  
  if (fallback.rows[0]) {
    return fallback.rows[0][`${phase}_script`] || `Welcome to Day ${day}, the ${phase} phase.`;
  }
  
  return `Welcome to Day ${day}, the ${phase} phase. Let me tell you something wonderful.`;
}

async function generateVideo(day, phase, script) {
  console.log(`  Generating Day ${day} ${phase}...`);
  
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
          talking_photo_id: AVATAR_ID
        },
        voice: {
          type: 'text',
          input_text: script.substring(0, 1500), // HeyGen limit
          voice_id: VOICE_ID
        }
      }],
      dimension: { width: 1280, height: 720 },
      test: false
    })
  });

  const data = await response.json();
  
  if (data.data?.video_id) {
    console.log(`  ✅ Video ID: ${data.data.video_id}`);
    
    // Save to database - simple INSERT
    try {
      await pool.query(`
        INSERT INTO heygen_videos (
          id, day_of_year, phase, age_category, archetype, 
          heygen_video_id, status, created_at, updated_at
        ) VALUES (
          gen_random_uuid(), $1, $2, 'adult', 'storyteller',
          $3, 'processing', NOW(), NOW()
        )
      `, [day, phase, data.data.video_id]);
    } catch (e) {
      // If duplicate, update instead
      await pool.query(`
        UPDATE heygen_videos 
        SET heygen_video_id = $3, status = 'processing', updated_at = NOW()
        WHERE day_of_year = $1 AND phase = $2 AND age_category = 'adult' AND archetype = 'storyteller'
      `, [day, phase, data.data.video_id]);
    }
    
    return data.data.video_id;
  } else {
    console.log(`  ❌ Error:`, data.error?.message || JSON.stringify(data));
    return null;
  }
}

async function main() {
  console.log('='.repeat(50));
  console.log(`URGENT: Generating Day ${TODAY} (TODAY'S LESSON)`);
  console.log('='.repeat(50));
  
  // Check credits first
  const quotaRes = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const quota = await quotaRes.json();
  const credits = quota.data?.remaining_quota || 0;
  console.log(`HeyGen credits: ${(credits/60).toFixed(0)} minutes`);
  
  if (credits < 300) { // Less than 5 minutes
    console.log('❌ Not enough credits!');
    await pool.end();
    return;
  }
  
  const videoIds = [];
  
  for (const phase of PHASES) {
    const script = await getScript(TODAY, phase);
    console.log(`  Script for ${phase}: ${script.substring(0, 50)}...`);
    
    const videoId = await generateVideo(TODAY, phase, script);
    if (videoId) {
      videoIds.push({ phase, videoId });
    }
    
    // Rate limit
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log('');
  console.log('='.repeat(50));
  console.log(`Generated ${videoIds.length}/${PHASES.length} videos for Day ${TODAY}`);
  console.log('Videos are processing on HeyGen (takes 2-5 minutes)');
  console.log('Run sync-completed-videos.cjs in 5 minutes to update URLs');
  console.log('='.repeat(50));
  
  // Save progress
  fs.writeFileSync('day34-progress.json', JSON.stringify({
    day: TODAY,
    timestamp: new Date().toISOString(),
    videoIds
  }, null, 2));
  
  await pool.end();
}

main().catch(console.error);
