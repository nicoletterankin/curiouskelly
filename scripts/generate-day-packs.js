/**
 * Generate skeleton day packs for all 365 days
 * Uses curriculum data to create basic lesson info for offline use
 * 
 * Usage: node scripts/generate-day-packs.js
 * Or: node scripts/generate-day-packs.js 17  (single day)
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DATA = path.join(__dirname, '..', 'public', 'data');
const CURRICULUM_LEARN = path.join(PUBLIC_DATA, 'curriculum', 'year1-foundations');
const CURRICULUM_GROW = path.join(PUBLIC_DATA, 'curriculum', 'year2-ai-fluency');

const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'];

// Load all curriculum data
function loadAllCurriculum() {
  const learn = {};
  const grow = {};
  
  for (const month of MONTHS) {
    const learnPath = path.join(CURRICULUM_LEARN, `${month}_curriculum.json`);
    const growPath = path.join(CURRICULUM_GROW, `${month}_curriculum.json`);
    
    if (fs.existsSync(learnPath)) {
      const data = JSON.parse(fs.readFileSync(learnPath, 'utf8'));
      for (const day of data.days || []) {
        learn[day.day] = day;
      }
    }
    
    if (fs.existsSync(growPath)) {
      const data = JSON.parse(fs.readFileSync(growPath, 'utf8'));
      for (const day of data.days || []) {
        grow[day.day] = day;
      }
    }
  }
  
  return { learn, grow };
}

// Generate a skeleton day pack
function generateDayPack(dayNumber, learnData, growData) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const topic = learnData?.title || `Day ${dayNumber} Lesson`;
  const objective = learnData?.learning_objective || 'Discover something new today.';
  
  const pack = {
    meta: {
      created_at: new Date().toISOString(),
      day_number: dayNumber,
      version: 'v3.0-skeleton',
      is_skeleton: true
    },
    lesson: {
      day_number: dayNumber,
      topic: topic,
      headline: objective,
      universal_truth: objective,
      emoji: learnData?.icon || '📚',
      category: learnData?.category || 'General',
      thumbnail_url: `/generated-visuals/day-${paddedDay}/thumbnail.png`
    },
    // Minimal atoms for basic playback
    atoms: [
      {
        id: `day${paddedDay}-hook-001`,
        phase: 'Hook',
        content: {
          script: `Welcome to Day ${dayNumber}! Today we are exploring: ${topic}. ${objective}`,
          kellyPose: 'welcome',
          kellyEmotion: 'curious'
        }
      },
      {
        id: `day${paddedDay}-wisdom-001`,
        phase: 'Wisdom',
        content: {
          script: objective,
          kellyPose: 'warm',
          kellyEmotion: 'gentle'
        }
      },
      {
        id: `day${paddedDay}-outro-001`,
        phase: 'Outro',
        content: {
          script: `That is today lesson! See you tomorrow for more learning adventures. Stay curious!`,
          kellyPose: 'wave',
          kellyEmotion: 'happy'
        }
      }
    ],
    // Grow track info
    grow: growData ? {
      topic: growData.title,
      objective: growData.learning_objective
    } : null,
    // Age variants with just the core message
    ageVariants: {
      '2-5': {
        persona: 'Playful Friend',
        phases: {
          hook: `Hi little friend! Today we are learning about ${topic.toLowerCase()}!`,
          wisdom: objective,
          outro: 'Great job learning today! See you tomorrow!'
        }
      },
      '6-12': {
        persona: 'Cool Big Sister',
        phases: {
          hook: `Hey curious one! Ready to discover something cool about ${topic.toLowerCase()}?`,
          wisdom: objective,
          outro: 'You are getting smarter every day. See you tomorrow!'
        }
      },
      '13-17': {
        persona: 'Smart Mentor',
        phases: {
          hook: `Let us explore: ${topic}. Here is what you need to know.`,
          wisdom: objective,
          outro: 'That is the real stuff. See you tomorrow.'
        }
      },
      '18-35': {
        persona: 'Equal Partner',
        phases: {
          hook: `Today topic: ${topic}. Let us break it down.`,
          wisdom: objective,
          outro: 'Knowledge that makes a difference. See you tomorrow.'
        }
      },
      '36-60': {
        persona: 'Respectful Guide',
        phases: {
          hook: `Today we explore ${topic} - a subject worth understanding.`,
          wisdom: objective,
          outro: 'Wisdom worth carrying forward. Until tomorrow.'
        }
      },
      '61-102': {
        persona: 'Honored Equal',
        phases: {
          hook: `Today we reflect on: ${topic}. A timeless subject.`,
          wisdom: objective,
          outro: 'Until we meet again. Take care.'
        }
      }
    }
  };
  
  return pack;
}

// Generate JS file content
function generateJsFile(dayNumber, pack) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  
  return `/**
 * Day ${paddedDay} Data Pack - "${pack.lesson.topic}"
 * ${pack.meta.is_skeleton ? 'SKELETON - Basic curriculum data only' : 'Complete lesson with all phases'}
 * Generated: ${new Date().toISOString()}
 */
window.CURIOUS_KELLY = window.CURIOUS_KELLY || {};
window.CURIOUS_KELLY.LOCAL_PACKS = window.CURIOUS_KELLY.LOCAL_PACKS || {};
window.CURIOUS_KELLY.DAY_${paddedDay} = ${JSON.stringify(pack, null, 2)};
window.CURIOUS_KELLY.LOCAL_PACKS[${dayNumber}] = window.CURIOUS_KELLY.DAY_${paddedDay};
`;
}

// Check if full pack already exists
function hasFullPack(dayNumber) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const packPath = path.join(PUBLIC_DATA, `day-${paddedDay}-complete.js`);
  
  if (!fs.existsSync(packPath)) return false;
  
  const content = fs.readFileSync(packPath, 'utf8');
  // Full packs have many atoms (Hook, Cliff, Fact1, Fact2, Fact3, Wisdom, Outro)
  return content.includes('"Fact1"') && content.includes('"Fact2"');
}

// Main
async function main() {
  const singleDay = parseInt(process.argv[2]);
  const { learn, grow } = loadAllCurriculum();
  
  console.log(`Loaded curriculum: ${Object.keys(learn).length} Learn days, ${Object.keys(grow).length} Grow days`);
  
  const daysToGenerate = singleDay ? [singleDay] : Array.from({length: 365}, (_, i) => i + 1);
  let generated = 0;
  let skipped = 0;
  
  for (const dayNumber of daysToGenerate) {
    // Skip if full pack already exists
    if (hasFullPack(dayNumber)) {
      skipped++;
      continue;
    }
    
    const learnData = learn[dayNumber];
    const growData = grow[dayNumber];
    
    if (!learnData) {
      console.warn(`No Learn curriculum for Day ${dayNumber}`);
      continue;
    }
    
    const pack = generateDayPack(dayNumber, learnData, growData);
    const jsContent = generateJsFile(dayNumber, pack);
    
    const paddedDay = String(dayNumber).padStart(3, '0');
    const outputPath = path.join(PUBLIC_DATA, `day-${paddedDay}-complete.js`);
    
    fs.writeFileSync(outputPath, jsContent);
    generated++;
  }
  
  console.log(`Generated ${generated} skeleton day packs`);
  console.log(`Skipped ${skipped} days (already have full packs)`);
}

main().catch(console.error);
