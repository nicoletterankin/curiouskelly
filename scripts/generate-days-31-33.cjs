#!/usr/bin/env node
/**
 * Generate Days 31-33 with REAL Curious Kelly
 * Using Adult Storyteller avatar
 */
require('dotenv').config();
const fs = require('fs');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const AVATAR_ID = '9ffd06bd986a4e3086612921f3ac87ea'; // Adult Storyteller
const VOICE_ID = 'BbuMXx40WT4ZuAgRXvNx';

// Simple scripts for Days 31-33 (placeholders - Antigravity should provide real ones)
const DAYS = {
  31: {
    topic: 'The Water Cycle',
    hook: "Have you ever wondered where rain comes from? Today we're exploring the incredible journey of water.",
    story: "Every drop of water you drink has been on an amazing journey. It may have once been in a dinosaur, a cloud, or an ancient ocean.",
    wonder: "What if you could trace the journey of a single water molecule through time? Where might it have been?",
    action: "Put a glass of water in the sun with plastic wrap on top. Watch condensation form - that's the water cycle in action!",
    wisdom: "Like water, we are all part of endless cycles. What we put into the world comes back to us."
  },
  32: {
    topic: 'Why We Sleep',
    hook: "Why do we spend a third of our lives unconscious? Sleep is one of the most mysterious things our bodies do.",
    story: "Scientists discovered that while we sleep, our brains wash themselves with fluid, clearing out toxins built up during the day.",
    wonder: "What happens in your brain during dreams? And why do we forget most of them when we wake up?",
    action: "Tonight, keep a dream journal by your bed. Write down anything you remember as soon as you wake up.",
    wisdom: "Rest is not laziness. It's how we prepare for our next chapter of growth."
  },
  33: {
    topic: 'How Plants Communicate',
    hook: "Did you know plants can talk to each other? They just don't use words.",
    story: "When a caterpillar attacks a plant, it releases chemicals into the air. Nearby plants 'smell' the warning and prepare their defenses.",
    wonder: "What other secret conversations might be happening in nature that we haven't discovered yet?",
    action: "Find two plants near each other. Gently touch one's leaves for a minute. Observe if the other plant reacts over the next few days.",
    wisdom: "Communication doesn't require words. Listening doesn't require ears. Pay attention to the silent messages around you."
  }
};

async function generateVideo(day, phase, script) {
  console.log(`  ${phase}...`);
  
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
    return { day, phase, videoId: data.data.video_id };
  } else {
    console.log(`    ❌ Error: ${data.error?.message}`);
    return { day, phase, error: data.error?.message };
  }
}

async function main() {
  console.log('='.repeat(60));
  console.log('GENERATING DAYS 31-33 WITH REAL CURIOUS KELLY');
  console.log('='.repeat(60));
  
  const allResults = [];
  
  for (const [dayStr, content] of Object.entries(DAYS)) {
    const day = parseInt(dayStr);
    console.log(`\nDay ${day}: ${content.topic}`);
    
    for (const phase of ['hook', 'story', 'wonder', 'action', 'wisdom']) {
      const result = await generateVideo(day, phase, content[phase]);
      if (result.videoId) {
        console.log(`    ✅ ${result.videoId}`);
      }
      allResults.push(result);
      await new Promise(r => setTimeout(r, 2000)); // Rate limit
    }
  }
  
  const success = allResults.filter(r => r.videoId);
  
  console.log('\n' + '='.repeat(60));
  console.log(`COMPLETE: ${success.length}/${allResults.length} videos`);
  
  fs.writeFileSync('days-31-33-videos.json', JSON.stringify({
    timestamp: new Date().toISOString(),
    avatarId: AVATAR_ID,
    videos: success
  }, null, 2));
  
  console.log('Saved to days-31-33-videos.json');
  console.log('Wait 5 minutes for HeyGen to render, then run sync script.');
}

main().catch(console.error);
