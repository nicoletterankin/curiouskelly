/**
 * COMPLETE Static Day Generator
 * 
 * Creates bulletproof static data files that contain EVERYTHING needed
 * for a lesson to play - no API calls required, no placeholders allowed.
 * 
 * Usage: npx tsx scripts/generate-complete-static-day.ts --day 353
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import 'dotenv/config';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing SUPABASE credentials');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

interface PhaseContent {
  script: string;
  options?: Array<{ text: string; letter: string; quality: string; response: string }>;
  kellyPose?: string;
  kellyEmotion?: string;
}

interface CompleteAtom {
  id: string;
  phase: string;
  archetype: string;
  content: PhaseContent;
  hd_video_url: string | null;
  visual_url: string | null;
}

interface CompleteLessonPack {
  meta: {
    created_at: string;
    day_number: number;
    version: string;
    is_complete: boolean;
    validation: {
      all_scripts_present: boolean;
      video_coverage: string;
      missing_videos: string[];
    };
  };
  lesson: {
    day_number: number;
    topic: string;
    headline: string;
    universal_truth: string;
    emoji: string;
    category: string;
  };
  atoms: CompleteAtom[];
  phases: Record<string, {
    script: string;
    options?: Array<{ text: string; letter: string; quality: string; response: string }>;
    hd_video_url: string | null;
    kellyPose?: string;
    kellyEmotion?: string;
  }>;
}

async function generateCompleteDay(dayNumber: number): Promise<void> {
  console.log(`\n🔧 Generating COMPLETE static file for Day ${dayNumber}...\n`);

  // 1. Fetch core lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();

  if (lessonError || !lesson) {
    console.error(`❌ Day ${dayNumber} not found in core_lessons`);
    process.exit(1);
  }

  console.log(`📚 Topic: ${lesson.topic}`);

  // 2. Fetch ALL atoms for this lesson (all archetypes)
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id)
    .order('phase');

  if (atomsError || !atoms?.length) {
    console.error(`❌ No atoms found for Day ${dayNumber}`);
    process.exit(1);
  }

  console.log(`📦 Found ${atoms.length} atoms`);

  // 3. Validate - check for placeholders
  const PHASES = ['Hook', 'Cliff', 'Fact1', 'Fact2', 'Fact3', 'Wisdom', 'Outro'];
  // MVP archetypes only - we don't need all 12
  const MVP_ARCHETYPES = ['The Explorer', 'The Rebel', 'The Scientist'];
  const archetypes = MVP_ARCHETYPES;
  
  let hasPlaceholders = false;
  let missingVideos: string[] = [];
  let videosPresent = 0;

  for (const archetype of archetypes) {
    for (const phase of PHASES) {
      const atom = atoms.find(a => a.archetype === archetype && a.phase === phase);
      if (!atom) {
        console.error(`❌ MISSING: ${archetype} - ${phase}`);
        hasPlaceholders = true;
        continue;
      }

      const script = atom.content?.script || atom.script || '';
      if (!script || script.includes('This is') && script.includes('content for')) {
        console.error(`❌ PLACEHOLDER: ${archetype} - ${phase}: "${script.substring(0, 50)}..."`);
        hasPlaceholders = true;
      }

      if (atom.hd_video_url) {
        videosPresent++;
      } else {
        missingVideos.push(`${archetype}-${phase}`);
      }
    }
  }

  const totalExpected = archetypes.length * PHASES.length;
  const videoCoverage = `${videosPresent}/${totalExpected}`;
  
  console.log(`\n📊 Validation:`);
  console.log(`   Scripts: ${hasPlaceholders ? '❌ HAS PLACEHOLDERS' : '✅ All real'}`);
  console.log(`   Videos: ${videoCoverage} (${Math.round(videosPresent/totalExpected*100)}%)`);
  
  if (missingVideos.length > 0 && missingVideos.length <= 10) {
    console.log(`   Missing: ${missingVideos.join(', ')}`);
  }

  if (hasPlaceholders) {
    console.error(`\n🚨 CANNOT GENERATE: Day ${dayNumber} has placeholder content!`);
    console.error(`   Fix the content in Supabase before generating static files.`);
    process.exit(1);
  }

  // 4. Build complete atoms array (MVP archetypes only)
  const mvpAtoms = atoms.filter(a => MVP_ARCHETYPES.includes(a.archetype));
  const completeAtoms: CompleteAtom[] = mvpAtoms.map(atom => ({
    id: atom.id,
    phase: atom.phase,
    archetype: atom.archetype,
    content: {
      script: atom.content?.script || atom.script || '',
      options: atom.content?.options || atom.options,
      kellyPose: atom.content?.kellyPose,
      kellyEmotion: atom.content?.kellyEmotion,
    },
    hd_video_url: atom.hd_video_url,
    visual_url: atom.visual_url,
  }));

  // 5. Build phases object (for backward compatibility)
  const phases: Record<string, any> = {};
  const explorerAtoms = atoms.filter(a => a.archetype === 'The Explorer');
  
  for (const phase of PHASES) {
    const atom = explorerAtoms.find(a => a.phase === phase);
    if (atom) {
      phases[phase] = {
        script: atom.content?.script || atom.script || '',
        options: atom.content?.options,
        hd_video_url: atom.hd_video_url,
        kellyPose: atom.content?.kellyPose || 'neutral',
        kellyEmotion: atom.content?.kellyEmotion || 'curious',
      };
    }
  }

  // 6. Build complete pack
  const pack: CompleteLessonPack = {
    meta: {
      created_at: new Date().toISOString(),
      day_number: dayNumber,
      version: 'v5.0-complete',
      is_complete: true,
      validation: {
        all_scripts_present: !hasPlaceholders,
        video_coverage: videoCoverage,
        missing_videos: missingVideos.slice(0, 20),
      },
    },
    lesson: {
      day_number: dayNumber,
      topic: lesson.topic,
      headline: lesson.marketing_headline || lesson.topic,
      universal_truth: lesson.universal_truth || '',
      emoji: lesson.emoji || '📚',
      category: lesson.category || 'general',
    },
    atoms: completeAtoms,
    phases,
  };

  // 7. Write file
  const outputDir = path.join(process.cwd(), 'public', 'data');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const filename = `day-${dayNumber}-complete.js`;
  const filepath = path.join(outputDir, filename);

  const jsContent = `/**
 * Day ${dayNumber} Data Pack - "${lesson.topic}"
 * COMPLETE - All content validated, no placeholders
 * Generated: ${new Date().toISOString()}
 */
window.CURIOUS_KELLY = window.CURIOUS_KELLY || {};
window.CURIOUS_KELLY.LOCAL_PACKS = window.CURIOUS_KELLY.LOCAL_PACKS || {};
window.CURIOUS_KELLY.DAY_${dayNumber} = ${JSON.stringify(pack, null, 2)};
`;

  fs.writeFileSync(filepath, jsContent);
  console.log(`\n✅ Generated: ${filepath}`);
  console.log(`   Size: ${(fs.statSync(filepath).size / 1024).toFixed(1)} KB`);
}

// Parse args
const args = process.argv.slice(2);
const dayIndex = args.indexOf('--day');
if (dayIndex === -1 || !args[dayIndex + 1]) {
  console.log('Usage: npx tsx scripts/generate-complete-static-day.ts --day 353');
  process.exit(1);
}

const dayNumber = parseInt(args[dayIndex + 1], 10);
generateCompleteDay(dayNumber).catch(console.error);
