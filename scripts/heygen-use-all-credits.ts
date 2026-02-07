#!/usr/bin/env npx tsx
/**
 * 🚨 HEYGEN CREDIT BURNER - USE ALL 600+ CREDITS TODAY
 * 
 * This script generates HeyGen videos for flagship days using ALL available credits.
 * 
 * Usage:
 *   npx tsx scripts/heygen-use-all-credits.ts --days 1-7
 *   npx tsx scripts/heygen-use-all-credits.ts --days 1-30 --ages adult
 *   npx tsx scripts/heygen-use-all-credits.ts --status
 */

import 'dotenv/config';
import { config } from 'dotenv';

config({ path: '.env.local' });
config({ path: '.env' });

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// 12 Archetypes with their adult talking photo IDs
const ADULT_AVATAR_IDS: Record<string, string> = {
  architect: "afc54d3abfc04947bec026b9ec917ce8",
  diplomat: "433ad96bf5d647d9964cecf784d008f6",
  empath: "aa8b5eb1d711468a9a6e2085a4f8469c",
  explorer: "45e5ef8b651846e0b62b7477e552e87b",
  macgyver: "b9032c922c6e4e35b58a98abd499d060",
  mystic: "a2b31ed0b5f84b0fa02d15d411735d3a",
  provider: "06b78109ad22489ea2165ebbf180f77b",
  rebel: "e614671b193c40f99772f7de5d1c51f7",
  scientist: "7bb18cddacd44333813cc90ffa44f766",
  storyteller: "9ffd06bd986a4e3086612921f3ac87ea",
  strategist: "2411df8bdb0d40b088aa453d4c2a2d20",
  survivor: "3f44bd33bfd1494d916d2746808a1a39"
};

const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

// Map day to archetype (based on lesson topic)
const DAY_TO_ARCHETYPE: Record<number, string> = {
  1: 'scientist',    // Magnets
  2: 'explorer',     // Outer space
  3: 'diplomat',     // Communication
  4: 'architect',    // Building
  5: 'storyteller',  // Myths
  6: 'mystic',       // Philosophy
  7: 'rebel',        // Innovation
  30: 'strategist',  // Planning
  100: 'provider',   // Community
  365: 'survivor',   // Resilience
};

async function checkCredits(): Promise<number> {
  console.log('Checking HeyGen credits...\n');
  
  if (!HEYGEN_API_KEY) {
    console.error('❌ HEYGEN_API_KEY not set');
    return 0;
  }
  
  try {
    const response = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    
    const data = await response.json();
    const credits = data?.data?.remaining_quota || 0;
    
    console.log(`✅ HeyGen Credits Remaining: ${credits}`);
    return credits;
  } catch (err: any) {
    console.error('Failed to check credits:', err.message);
    return 0;
  }
}

async function generateVideo(params: {
  day: number;
  phase: string;
  archetype: string;
  script: string;
}): Promise<{ videoId: string } | null> {
  const avatarId = ADULT_AVATAR_IDS[params.archetype];
  if (!avatarId) {
    console.error(`Unknown archetype: ${params.archetype}`);
    return null;
  }
  
  console.log(`  Submitting: Day ${params.day} / ${params.phase} / ${params.archetype}`);
  
  try {
    // First get a voice ID
    const voicesRes = await fetch('https://api.heygen.com/v2/voices', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY!, Accept: 'application/json' }
    });
    const voicesData = await voicesRes.json();
    const voices = voicesData?.data?.voices || [];
    const femaleVoice = voices.find((v: any) => 
      v.language?.toLowerCase().includes('english') && 
      v.gender?.toLowerCase() === 'female'
    ) || voices[0];
    const voiceId = femaleVoice?.voice_id;
    
    if (!voiceId) {
      console.error('  ❌ No voice found');
      return null;
    }
    
    // Submit video generation
    const payload = {
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: avatarId
        },
        voice: {
          type: 'text',
          voice_id: voiceId,
          input_text: params.script
        },
        background: { type: 'color', value: '#FFFFFF' }
      }],
      dimension: { width: 1080, height: 1920 },
      test: false
    };
    
    const response = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY!,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    const videoId = data?.data?.video_id;
    
    if (videoId) {
      console.log(`  ✅ Submitted: ${videoId}`);
      return { videoId };
    } else {
      console.error(`  ❌ No video ID returned:`, data);
      return null;
    }
  } catch (err: any) {
    console.error(`  ❌ Error: ${err.message}`);
    return null;
  }
}

async function getScript(day: number, phase: string): Promise<string> {
  // For now, return a placeholder script
  // In production, this would fetch from the database
  const scripts: Record<string, Record<string, string>> = {
    '1': {
      hook: "Have you ever wondered why magnets stick to your refrigerator? Today we're going to discover the invisible force that makes this magic happen!",
      story: "Thousands of years ago, people in ancient Greece found a strange rock that could attract iron. They called it magnetite, and it would change the world forever.",
      wonder: "Here's something amazing - the Earth itself is a giant magnet! That's why compasses always point north.",
      action: "Try this: take a magnet and slowly move it toward different objects. Which ones does it attract? Make a list!",
      wisdom: "Magnets remind us that some of the most powerful forces in the universe are invisible. What other invisible forces can you think of?"
    }
  };
  
  return scripts[String(day)]?.[phase] || 
    `This is the ${phase} phase for day ${day}. Kelly is teaching about an amazing topic that will change how you see the world.`;
}

async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--status') || args.includes('-s')) {
    await checkCredits();
    return;
  }
  
  console.log('🚀 HEYGEN CREDIT BURNER\n');
  console.log('='.repeat(50));
  
  // Check credits first
  const credits = await checkCredits();
  if (credits < 5) {
    console.error('\n❌ Not enough credits to proceed');
    return;
  }
  
  // Parse day range
  let days: number[] = [1, 2, 3, 4, 5, 6, 7]; // Default: first week
  
  const daysArg = args.find(a => a.startsWith('--days='));
  if (daysArg) {
    const range = daysArg.split('=')[1];
    if (range.includes('-')) {
      const [start, end] = range.split('-').map(Number);
      days = Array.from({ length: end - start + 1 }, (_, i) => i + start);
    } else {
      days = range.split(',').map(Number);
    }
  }
  
  console.log(`\nDays to process: ${days.join(', ')}`);
  console.log(`Phases per day: ${PHASES.length}`);
  console.log(`Total videos: ${days.length * PHASES.length}`);
  console.log(`Estimated credits: ${days.length * PHASES.length}`);
  console.log('='.repeat(50) + '\n');
  
  const results: any[] = [];
  let submitted = 0;
  
  for (const day of days) {
    const archetype = DAY_TO_ARCHETYPE[day] || 'scientist';
    console.log(`\n📅 Day ${day} (${archetype})`);
    
    for (const phase of PHASES) {
      const script = await getScript(day, phase);
      
      const result = await generateVideo({
        day,
        phase,
        archetype,
        script
      });
      
      if (result) {
        results.push({
          day,
          phase,
          archetype,
          videoId: result.videoId,
          status: 'submitted'
        });
        submitted++;
      }
      
      // Rate limit: 5 seconds between submissions
      await new Promise(r => setTimeout(r, 5000));
    }
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('📊 SUMMARY');
  console.log('='.repeat(50));
  console.log(`Submitted: ${submitted} videos`);
  console.log(`\nVideo IDs saved to: heygen-jobs-${new Date().toISOString().split('T')[0]}.json`);
  
  // Save results
  const fs = await import('fs');
  fs.writeFileSync(
    `heygen-jobs-${new Date().toISOString().split('T')[0]}.json`,
    JSON.stringify(results, null, 2)
  );
  
  console.log('\n⏰ Videos are processing on HeyGen. Check status in ~5-10 minutes.');
  console.log('Run: npx tsx scripts/heygen-check-status.ts');
}

main().catch(console.error);
