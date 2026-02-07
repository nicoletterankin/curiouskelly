#!/usr/bin/env node
/**
 * HeyGen Batch Video Generator - Days 1-30, All Phases
 * 
 * Uses 668.5 credits to generate lip-synced Kelly videos
 * Run: node scripts/heygen-batch-day1-30.js
 */

import { config } from 'dotenv';
import fs from 'fs';

config();
config({ path: '.env.local' });

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// Storyteller avatar (adult)
const STORYTELLER_AVATAR = '9ffd06bd986a4e3086612921f3ac87ea';

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

// Sample scripts for each phase (to be replaced with real content)
const PHASE_SCRIPTS = {
  hook: "Welcome to today's lesson! Let's spark your curiosity with something fascinating.",
  story: "Let me tell you an incredible story that will change how you see the world.",
  wonder: "Now let's explore the deeper questions this brings up. What do you wonder about?",
  action: "Time to put this into practice! Here's something you can try right now.",
  wisdom: "Before we go, let me share one important insight to carry with you today."
};

// Track progress
const progress = {
  started: new Date().toISOString(),
  videos: [],
  errors: [],
  creditsStart: 0,
  creditsEnd: 0
};

async function checkCredits() {
  const res = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const data = await res.json();
  return data.data?.remaining_quota || 0;
}

async function generateVideo(day, phase) {
  const script = PHASE_SCRIPTS[phase];
  
  const res = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: STORYTELLER_AVATAR
        },
        voice: {
          type: 'text',
          input_text: script,
          voice_id: '1bd001e7e50f421d891986aad5158bc8'
        }
      }],
      dimension: { width: 1920, height: 1080 },
      test: false,
      title: `Day ${day} - ${phase.toUpperCase()}`
    })
  });
  
  return res.json();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  console.log('🚀 HeyGen Batch Generator - Days 1-30');
  console.log('=====================================\n');
  
  progress.creditsStart = await checkCredits();
  console.log(`💰 Starting credits: ${progress.creditsStart} seconds (${(progress.creditsStart/60).toFixed(1)} min)\n`);
  
  const startDay = 1;
  const endDay = 30;
  
  for (let day = startDay; day <= endDay; day++) {
    console.log(`\n📅 Day ${day}:`);
    
    for (const phase of PHASES) {
      process.stdout.write(`  ${phase.padEnd(8)}: `);
      
      try {
        const result = await generateVideo(day, phase);
        
        if (result.data?.video_id) {
          console.log(`✅ ${result.data.video_id}`);
          progress.videos.push({
            day,
            phase,
            videoId: result.data.video_id,
            timestamp: new Date().toISOString()
          });
        } else {
          console.log(`❌ ${result.error?.message || 'Unknown error'}`);
          progress.errors.push({ day, phase, error: result });
        }
        
        // Rate limit: wait 1 second between requests
        await sleep(1000);
        
      } catch (err) {
        console.log(`❌ ${err.message}`);
        progress.errors.push({ day, phase, error: err.message });
      }
    }
    
    // Every 5 days, check remaining credits
    if (day % 5 === 0) {
      const remaining = await checkCredits();
      console.log(`\n  💰 Credits remaining: ${remaining} seconds (${(remaining/60).toFixed(1)} min)`);
    }
  }
  
  // Final stats
  progress.creditsEnd = await checkCredits();
  const used = progress.creditsStart - progress.creditsEnd;
  
  console.log('\n\n=====================================');
  console.log('📊 BATCH COMPLETE');
  console.log('=====================================');
  console.log(`Videos generated: ${progress.videos.length}`);
  console.log(`Errors: ${progress.errors.length}`);
  console.log(`Credits used: ${used} seconds (${(used/60).toFixed(1)} min)`);
  console.log(`Credits remaining: ${progress.creditsEnd} seconds (${(progress.creditsEnd/60).toFixed(1)} min)`);
  
  // Save progress to file
  fs.writeFileSync(
    'heygen-batch-progress.json',
    JSON.stringify(progress, null, 2)
  );
  console.log('\n💾 Progress saved to heygen-batch-progress.json');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
