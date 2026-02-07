#!/usr/bin/env node
require('dotenv').config();
const fs = require('fs');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// CORRECT IDs from kelly-assets.ts
const AVATAR_ID = '5e5796ea458b4a5fa5b698c9b51dbc8d'; // base_adult talking photo
const VOICE_ID = 'BbuMXx40WT4ZuAgRXvNx'; // adult HeyGen voice

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
    return { phase, videoId: data.data.video_id };
  } else {
    console.log(`  ❌ ${phase}: ${data.error?.message || JSON.stringify(data)}`);
    return { phase, error: data.error?.message || 'Unknown' };
  }
}

async function main() {
  console.log('DAY 34 - CORRECT VOICE ID');
  console.log('Avatar:', AVATAR_ID);
  console.log('Voice:', VOICE_ID);
  console.log('='.repeat(40));
  
  const results = [];
  
  for (const phase of ['hook', 'story', 'wonder', 'action', 'wisdom']) {
    const result = await generateVideo(phase, SCRIPTS[phase]);
    results.push(result);
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log('');
  console.log('RESULTS:');
  const success = results.filter(r => r.videoId);
  success.forEach(r => console.log(`  ${r.phase}: ${r.videoId}`));
  
  if (success.length > 0) {
    fs.writeFileSync('day34-correct-videos.json', JSON.stringify({
      day: 34,
      timestamp: new Date().toISOString(),
      avatarId: AVATAR_ID,
      voiceId: VOICE_ID,
      videos: success
    }, null, 2));
    console.log('');
    console.log('Saved to day34-correct-videos.json');
    console.log('HeyGen takes 2-5 min to render. Then run sync script.');
  }
}

main().catch(console.error);
