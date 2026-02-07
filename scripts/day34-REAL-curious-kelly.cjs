#!/usr/bin/env node
/**
 * DAY 34 - REAL CURIOUS KELLY (Custom Talking Photo)
 * Using the CORRECT avatar IDs from heygen_talking_photo_ids.json
 */
require('dotenv').config();
const fs = require('fs');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// REAL CURIOUS KELLY - Adult Storyteller
// From: generated-images/kelly-archetypes-head-only/age/adult/heygen_talking_photo_ids.json
const AVATAR_ID = '9ffd06bd986a4e3086612921f3ac87ea'; // storyteller
const VOICE_ID = 'BbuMXx40WT4ZuAgRXvNx'; // Kelly2 voice

// Day 34 - How Magnets Work
const SCRIPTS = {
  hook: "What makes magnets attract and repel? Today we're exploring the invisible forces that shape our world.",
  story: "Long ago, ancient Greeks discovered lodestone, a rock that could move iron. They thought it was magic, but it was actually the first discovery of magnetism.",
  wonder: "What other invisible forces might be shaping our universe right now? Scientists are still discovering new things about magnetism every day.",
  action: "Try this at home: Take a magnet and test different objects. Make a list of what's magnetic and what isn't. You might be surprised by what you find!",
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
          type: 'talking_photo',
          talking_photo_id: AVATAR_ID
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
  console.log('='.repeat(60));
  console.log('DAY 34 - REAL CURIOUS KELLY');
  console.log('Avatar: Adult Storyteller (9ffd06bd986a4e3086612921f3ac87ea)');
  console.log('Topic: How Magnets Work');
  console.log('='.repeat(60));
  
  // Check credits first
  const quotaRes = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const quota = await quotaRes.json();
  console.log(`Credits: ${(quota.data?.remaining_quota/60).toFixed(0)} minutes remaining`);
  console.log('');
  
  const results = [];
  
  for (const phase of ['hook', 'story', 'wonder', 'action', 'wisdom']) {
    const result = await generateVideo(phase, SCRIPTS[phase]);
    results.push(result);
    await new Promise(r => setTimeout(r, 2000));
  }
  
  const success = results.filter(r => r.videoId);
  
  console.log('');
  console.log('='.repeat(60));
  console.log(`SUCCESS: ${success.length}/5 videos`);
  success.forEach(r => console.log(`  ${r.phase}: ${r.videoId}`));
  
  fs.writeFileSync('day34-curious-kelly-videos.json', JSON.stringify({
    day: 34,
    topic: 'How Magnets Work',
    date: 'February 3, 2026',
    avatar: 'Adult Storyteller',
    avatarId: AVATAR_ID,
    timestamp: new Date().toISOString(),
    videos: success
  }, null, 2));
  
  console.log('');
  console.log('Saved to day34-curious-kelly-videos.json');
  console.log('Wait 3-5 minutes for HeyGen to render, then run sync script.');
}

main().catch(console.error);
