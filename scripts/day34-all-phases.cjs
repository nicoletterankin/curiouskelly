#!/usr/bin/env node
require('dotenv').config();
const fs = require('fs');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const AVATAR_ID = '9ffd06bd986a4e3086612921f3ac87ea';
const VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku';

// Day 34 scripts (popcorn/magnets topic)
const SCRIPTS = {
  hook: "What makes popcorn explode into fluffy shapes? Today we're exploring the science of magnetism and how invisible forces shape our world.",
  story: "Long ago, ancient Greeks discovered a strange rock called lodestone. When they placed it near iron, the iron moved on its own. They thought it was magic, but it was actually the first discovery of magnetism.",
  wonder: "What else might we discover about how invisible forces connect everything in our universe? Scientists are still finding new magnetic phenomena today.",
  action: "Try this at home: Take a magnet and see what objects it attracts. Make a list of magnetic and non-magnetic items. You might be surprised by what you find!",
  wisdom: "The best way to predict the future is to understand the forces that shape it. Magnetism teaches us that the most powerful forces are often the ones we cannot see."
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
        character: { type: 'talking_photo', talking_photo_id: AVATAR_ID },
        voice: { type: 'text', input_text: script, voice_id: VOICE_ID }
      }],
      dimension: { width: 1280, height: 720 }
    })
  });

  const data = await response.json();
  
  if (data.data?.video_id) {
    console.log(`  ✅ ${phase}: ${data.data.video_id}`);
    return { phase, videoId: data.data.video_id, status: 'processing' };
  } else {
    console.log(`  ❌ ${phase}: ${data.error?.message || 'Unknown error'}`);
    return { phase, error: data.error?.message };
  }
}

async function main() {
  console.log('DAY 34 - ALL 5 PHASES');
  console.log('=====================');
  
  // Already have hook generating
  const results = [
    { phase: 'hook', videoId: 'e7b26ffc45174596bd55ba0650437085', status: 'processing' }
  ];
  console.log('hook: Already generating (e7b26ffc...)');
  
  // Generate remaining
  for (const phase of ['story', 'wonder', 'action', 'wisdom']) {
    const result = await generateVideo(phase, SCRIPTS[phase]);
    results.push(result);
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log('');
  console.log('SUMMARY:');
  results.forEach(r => {
    if (r.videoId) console.log(`  ${r.phase}: ${r.videoId}`);
    else console.log(`  ${r.phase}: FAILED`);
  });
  
  fs.writeFileSync('day34-videos.json', JSON.stringify({
    day: 34,
    timestamp: new Date().toISOString(),
    videos: results.filter(r => r.videoId)
  }, null, 2));
  
  console.log('');
  console.log('Videos processing on HeyGen. Run sync in 5 minutes.');
}

main().catch(console.error);
