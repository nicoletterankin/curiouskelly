#!/usr/bin/env npx tsx
/**
 * 🎙️ FULL LESSON AUDIO GENERATOR
 * 
 * Generates ALL audio clips for a complete lesson experience:
 * - 7 intro clips (one per phase)
 * - 17 response clips (all options across all phases)
 * = 24 total clips per archetype per day
 * 
 * This script works with the GOLD STANDARD content structure.
 * 
 * Usage:
 *   npx tsx scripts/generate-full-lesson-audio.ts --day=355 --archetype="The Explorer"
 *   npx tsx scripts/generate-full-lesson-audio.ts --day=355 --all-archetypes
 *   npx tsx scripts/generate-full-lesson-audio.ts --from-gold-standard=content/gold-standard/DAY-355-EXPLORER-LOCKED.json
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  
  // Kelly's voice ID in ElevenLabs
  KELLY_VOICE_ID: process.env.ELEVENLABS_VOICE_ID || process.env.KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0', // Kelly voice
  
  // Output directory
  OUTPUT_DIR: path.join(process.cwd(), 'generated-audio', 'full-lessons'),
  
  // Rate limiting
  RATE_LIMIT_MS: 500, // 500ms between API calls
};

// =============================================================================
// TYPES
// =============================================================================

interface ClipSpec {
  clipId: string;
  script: string;
  phase: string;
  type: 'intro' | 'response';
  optionLetter?: string;
}

interface GenerationResult {
  clipId: string;
  success: boolean;
  audioPath?: string;
  duration?: number;
  error?: string;
}

// =============================================================================
// ELEVENLABS API
// =============================================================================

async function generateAudio(text: string, clipId: string): Promise<GenerationResult> {
  try {
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        },
        body: JSON.stringify({
          text: text,
          model_id: 'eleven_turbo_v2_5',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.3,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ElevenLabs API error: ${response.status} - ${errorText}`);
    }

    const audioBuffer = await response.arrayBuffer();
    const audioPath = path.join(CONFIG.OUTPUT_DIR, `${clipId}.mp3`);
    
    fs.writeFileSync(audioPath, Buffer.from(audioBuffer));
    
    // Estimate duration (roughly 150 words per minute, average 5 chars per word)
    const wordCount = text.split(/\s+/).length;
    const estimatedDuration = Math.round((wordCount / 150) * 60);
    
    return {
      clipId,
      success: true,
      audioPath,
      duration: estimatedDuration,
    };
  } catch (error: any) {
    return {
      clipId,
      success: false,
      error: error.message,
    };
  }
}

// =============================================================================
// EXTRACT CLIPS FROM LESSON ATOM
// =============================================================================

function extractClipsFromAtom(atom: any): ClipSpec[] {
  const clips: ClipSpec[] = [];
  const content = atom.content;
  
  // 1. Main intro script
  if (content.script && content.clipId) {
    clips.push({
      clipId: content.clipId,
      script: content.script,
      phase: atom.phase,
      type: 'intro',
    });
  }
  
  // 2. All option responses
  if (content.options && Array.isArray(content.options)) {
    for (const option of content.options) {
      if (option.response && option.responseClipId) {
        clips.push({
          clipId: option.responseClipId,
          script: option.response,
          phase: atom.phase,
          type: 'response',
          optionLetter: option.letter,
        });
      }
    }
  }
  
  return clips;
}

// =============================================================================
// MAIN GENERATION FUNCTION
// =============================================================================

async function generateFullLessonAudio(day: number, archetype: string): Promise<void> {
  const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🎙️ FULL LESSON AUDIO GENERATOR                                  ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Day: ${day}`.padEnd(67) + '║');
  console.log(`║  Archetype: ${archetype}`.padEnd(67) + '║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  // Ensure output directory exists
  if (!fs.existsSync(CONFIG.OUTPUT_DIR)) {
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  }

  // Get the lesson
  const { data: lessons, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', day);

  if (lessonError || !lessons?.length) {
    console.error(`❌ No lesson found for day ${day}`);
    return;
  }

  // Get all atoms for this archetype
  let allAtoms: any[] = [];
  for (const lesson of lessons) {
    const { data: atoms, error: atomsError } = await supabase
      .from('lesson_atoms')
      .select('*')
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', archetype);

    if (!atomsError && atoms) {
      allAtoms = allAtoms.concat(atoms);
    }
  }

  if (allAtoms.length === 0) {
    console.error(`❌ No atoms found for ${archetype} on day ${day}`);
    return;
  }

  console.log(`📚 Found ${allAtoms.length} phases\n`);

  // Extract all clips
  const allClips: ClipSpec[] = [];
  for (const atom of allAtoms) {
    const clips = extractClipsFromAtom(atom);
    allClips.push(...clips);
  }

  console.log(`🎬 Total clips to generate: ${allClips.length}`);
  console.log('─'.repeat(60));

  // Categorize clips
  const introClips = allClips.filter(c => c.type === 'intro');
  const responseClips = allClips.filter(c => c.type === 'response');
  
  console.log(`   📢 Intro clips: ${introClips.length}`);
  console.log(`   💬 Response clips: ${responseClips.length}`);
  console.log('─'.repeat(60) + '\n');

  // Generate each clip
  const results: GenerationResult[] = [];
  let successCount = 0;
  let failCount = 0;

  for (let i = 0; i < allClips.length; i++) {
    const clip = allClips[i];
    const progress = `[${i + 1}/${allClips.length}]`;
    
    console.log(`${progress} Generating ${clip.clipId}...`);
    
    const result = await generateAudio(clip.script, clip.clipId);
    results.push(result);
    
    if (result.success) {
      successCount++;
      console.log(`   ✅ Success (${result.duration}s)`);
    } else {
      failCount++;
      console.log(`   ❌ Failed: ${result.error}`);
    }
    
    // Rate limiting
    if (i < allClips.length - 1) {
      await new Promise(resolve => setTimeout(resolve, CONFIG.RATE_LIMIT_MS));
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('🏁 GENERATION COMPLETE');
  console.log('═'.repeat(60));
  console.log(`✅ Success: ${successCount}/${allClips.length}`);
  console.log(`❌ Failed: ${failCount}/${allClips.length}`);
  console.log(`📁 Output: ${CONFIG.OUTPUT_DIR}`);

  // Save manifest
  const manifest = {
    day,
    archetype,
    generatedAt: new Date().toISOString(),
    totalClips: allClips.length,
    successCount,
    failCount,
    clips: results,
  };

  const manifestPath = path.join(CONFIG.OUTPUT_DIR, `manifest-${day}-${archetype.replace(/\s+/g, '-').toLowerCase()}.json`);
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`📋 Manifest: ${manifestPath}`);
}

// =============================================================================
// FROM GOLD STANDARD FILE
// =============================================================================

async function generateFromGoldStandard(filePath: string): Promise<void> {
  console.log(`\n📄 Loading gold standard from: ${filePath}\n`);
  
  const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  const { meta, phases } = content;
  
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🎙️ GOLD STANDARD AUDIO GENERATOR                                ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Day: ${meta.day}`.padEnd(67) + '║');
  console.log(`║  Archetype: ${meta.archetype}`.padEnd(67) + '║');
  console.log(`║  Topic: ${meta.topic}`.padEnd(67) + '║');
  console.log(`║  Total Clips: ${meta.totalClips}`.padEnd(67) + '║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  // Ensure output directory exists
  const outputDir = path.join(CONFIG.OUTPUT_DIR, `day-${meta.day}`);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Extract all clips from gold standard
  const clips: ClipSpec[] = [];
  
  for (const [phaseName, phase] of Object.entries(phases) as [string, any][]) {
    // Intro clip
    if (phase.intro) {
      clips.push({
        clipId: phase.intro.clipId,
        script: phase.intro.script,
        phase: phaseName,
        type: 'intro',
      });
    }
    
    // Response clips
    if (phase.options) {
      for (const option of phase.options) {
        if (option.response) {
          clips.push({
            clipId: option.response.clipId,
            script: option.response.script,
            phase: phaseName,
            type: 'response',
            optionLetter: option.letter,
          });
        }
      }
    }
  }

  console.log(`🎬 Clips to generate: ${clips.length}`);
  console.log('─'.repeat(60));

  // Generate each clip
  const results: GenerationResult[] = [];
  let successCount = 0;

  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i];
    const progress = `[${i + 1}/${clips.length}]`;
    
    process.stdout.write(`${progress} ${clip.clipId}... `);
    
    // Override output path for this run
    const originalOutputDir = CONFIG.OUTPUT_DIR;
    CONFIG.OUTPUT_DIR = outputDir;
    
    const result = await generateAudio(clip.script, clip.clipId);
    results.push(result);
    
    CONFIG.OUTPUT_DIR = originalOutputDir;
    
    if (result.success) {
      successCount++;
      console.log(`✅ (${result.duration}s)`);
    } else {
      console.log(`❌ ${result.error}`);
    }
    
    // Rate limiting
    if (i < clips.length - 1) {
      await new Promise(resolve => setTimeout(resolve, CONFIG.RATE_LIMIT_MS));
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('🏁 GENERATION COMPLETE');
  console.log('═'.repeat(60));
  console.log(`✅ Success: ${successCount}/${clips.length}`);
  console.log(`📁 Output: ${outputDir}`);

  // Save manifest
  const manifest = {
    source: filePath,
    day: meta.day,
    archetype: meta.archetype,
    topic: meta.topic,
    generatedAt: new Date().toISOString(),
    totalClips: clips.length,
    successCount,
    clips: results,
  };

  const manifestPath = path.join(outputDir, 'generation-manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`📋 Manifest: ${manifestPath}`);
}

// =============================================================================
// CLI
// =============================================================================

const args = process.argv.slice(2);
const dayArg = args.find(a => a.startsWith('--day='));
const archetypeArg = args.find(a => a.startsWith('--archetype='));
const goldStandardArg = args.find(a => a.startsWith('--from-gold-standard='));

if (goldStandardArg) {
  const filePath = goldStandardArg.split('=')[1];
  generateFromGoldStandard(filePath).catch(console.error);
} else if (dayArg && archetypeArg) {
  const day = parseInt(dayArg.split('=')[1]);
  const archetype = archetypeArg.split('=')[1];
  generateFullLessonAudio(day, archetype).catch(console.error);
} else {
  console.log(`
Usage:
  npx tsx scripts/generate-full-lesson-audio.ts --day=355 --archetype="The Explorer"
  npx tsx scripts/generate-full-lesson-audio.ts --from-gold-standard=content/gold-standard/DAY-355-EXPLORER-LOCKED.json
  `);
}
