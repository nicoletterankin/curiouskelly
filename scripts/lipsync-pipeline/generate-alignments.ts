/**
 * Kelly Lipsync Pipeline - Alignment Generation
 * 
 * Processes generated audio files to create phoneme alignments.
 * Uses Montreal Forced Aligner when available, falls back to estimation.
 * Generates blendshape timelines for direct lipsync playback.
 * 
 * Usage:
 *   npx ts-node scripts/lipsync-pipeline/generate-alignments.ts
 *   npx ts-node scripts/lipsync-pipeline/generate-alignments.ts --input ./generated-audio
 */

import { createClient } from '@supabase/supabase-js';
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load both .env and .env.local from project root
const projectRoot = process.cwd();
dotenv.config({ path: path.join(projectRoot, '.env') });
dotenv.config({ path: path.join(projectRoot, '.env.local') });

// =============================================================================
// CONFIGURATION
// =============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const INPUT_DIR = path.join(process.cwd(), 'generated-audio');
const OUTPUT_DIR = path.join(process.cwd(), 'generated-alignments');

const FPS = 30; // Frames per second for blendshape timeline

// =============================================================================
// PHONEME TO VISEME MAPPING
// =============================================================================

const PHONEME_TO_VISEME: Record<string, string> = {
  // Vowels
  'AA': 'A', 'AE': 'A', 'AH': 'A',
  'AO': 'O', 'AW': 'O', 'OW': 'O', 'OY': 'O',
  'EH': 'E', 'EY': 'E',
  'IH': 'I', 'IY': 'I',
  'UH': 'U', 'UW': 'U', 'ER': 'R',
  'AY': 'A',
  
  // Consonants
  'P': 'M', 'B': 'M', 'M': 'M',
  'F': 'F', 'V': 'F',
  'TH': 'C', 'DH': 'C',
  'T': 'C', 'D': 'C', 'N': 'C', 'S': 'C', 'Z': 'C',
  'SH': 'SH', 'ZH': 'SH', 'CH': 'SH', 'JH': 'SH',
  'K': 'C', 'G': 'C', 'NG': 'C', 'HH': 'A',
  'L': 'L', 'R': 'R',
  'W': 'U', 'Y': 'I',
  
  // Silence
  'SIL': 'REST', 'SP': 'REST', 'spn': 'REST',
};

const VISEME_TO_BLENDSHAPES: Record<string, Record<string, number>> = {
  'A': { jawOpen: 80, mouthOpen: 75, mouthStretchLeft: 10, mouthStretchRight: 10 },
  'E': { jawOpen: 35, mouthOpen: 30, mouthStretchLeft: 35, mouthStretchRight: 35, mouthSmileLeft: 25, mouthSmileRight: 25 },
  'I': { jawOpen: 15, mouthOpen: 10, mouthStretchLeft: 50, mouthStretchRight: 50, mouthSmileLeft: 30, mouthSmileRight: 30 },
  'O': { jawOpen: 55, mouthOpen: 50, mouthFunnel: 45, mouthPucker: 20 },
  'U': { jawOpen: 20, mouthOpen: 15, mouthFunnel: 60, mouthPucker: 70 },
  'M': { jawOpen: 0, mouthClose: 100, mouthPressLeft: 40, mouthPressRight: 40 },
  'F': { jawOpen: 8, mouthOpen: 5, mouthUpperUpLeft: 30, mouthUpperUpRight: 30, mouthLowerDownLeft: 20, mouthLowerDownRight: 20 },
  'C': { jawOpen: 10, mouthOpen: 5, mouthStretchLeft: 25, mouthStretchRight: 25 },
  'L': { jawOpen: 18, mouthOpen: 12, mouthStretchLeft: 25, mouthStretchRight: 25 },
  'R': { jawOpen: 22, mouthOpen: 16, mouthFunnel: 30, mouthPucker: 20 },
  'SH': { jawOpen: 12, mouthOpen: 8, mouthFunnel: 50, mouthPucker: 40 },
  'REST': { jawOpen: 0, mouthOpen: 0, mouthSmileLeft: 15, mouthSmileRight: 15, mouthClose: 20 },
};

// Letter to phoneme estimation
const LETTER_TO_PHONEME: Record<string, string> = {
  'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AA', 'u': 'AH',
  'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
  'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
  'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
  't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z',
};

// =============================================================================
// TYPES
// =============================================================================

interface WordAlignment {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

interface PhoneAlignment {
  phone: string;
  start: number;
  end: number;
  word: string;
  viseme: string;
}

interface BlendshapeFrame {
  timestamp: number;
  blendshapes: Record<string, number>;
}

interface AlignmentResult {
  words: WordAlignment[];
  phones: PhoneAlignment[];
  duration: number;
  method: string;
  confidence: number;
  blendshapeTimeline?: BlendshapeFrame[];
}

interface ManifestEntry {
  day: number;
  ageBucket: string;
  language: string;
  phase: string;
  transcript: string;
  audioPath: string;
  shardId: string;
  duration?: number;
}

// =============================================================================
// ALIGNMENT FUNCTIONS
// =============================================================================

/**
 * Estimate alignment from transcript only (fallback method)
 */
function estimateAlignment(transcript: string, duration?: number): AlignmentResult {
  const words = transcript.trim().split(/\s+/).filter(w => w.length > 0);
  
  if (words.length === 0) {
    return {
      words: [],
      phones: [],
      duration: 0,
      method: 'estimation',
      confidence: 0,
    };
  }

  const estimatedDuration = duration || words.length * 0.4;
  const wordDuration = estimatedDuration / words.length;
  
  const wordAlignments: WordAlignment[] = [];
  const phoneAlignments: PhoneAlignment[] = [];
  let currentTime = 0;

  for (const word of words) {
    const wordEnd = currentTime + wordDuration;
    
    wordAlignments.push({
      word,
      start: Math.round(currentTime * 1000) / 1000,
      end: Math.round(wordEnd * 1000) / 1000,
      confidence: 0.5,
    });

    const phonemes = estimatePhonemesFromWord(word);
    const phoneDuration = wordDuration / phonemes.length;
    
    for (let i = 0; i < phonemes.length; i++) {
      const phone = phonemes[i];
      phoneAlignments.push({
        phone,
        start: Math.round((currentTime + (i * phoneDuration)) * 1000) / 1000,
        end: Math.round((currentTime + ((i + 1) * phoneDuration)) * 1000) / 1000,
        word,
        viseme: PHONEME_TO_VISEME[phone] || 'REST',
      });
    }

    currentTime = wordEnd + 0.03;
  }

  return {
    words: wordAlignments,
    phones: phoneAlignments,
    duration: estimatedDuration,
    method: 'estimation',
    confidence: 0.5,
  };
}

/**
 * Estimate phonemes from word spelling
 */
function estimatePhonemesFromWord(word: string): string[] {
  const phonemes: string[] = [];
  const lowerWord = word.toLowerCase().replace(/[^a-z]/g, '');
  
  for (let i = 0; i < lowerWord.length; i++) {
    const char = lowerWord[i];
    
    // Handle digraphs
    if (i < lowerWord.length - 1) {
      const digraph = lowerWord.substring(i, i + 2);
      if (digraph === 'th') { phonemes.push('TH'); i++; continue; }
      if (digraph === 'sh') { phonemes.push('SH'); i++; continue; }
      if (digraph === 'ch') { phonemes.push('CH'); i++; continue; }
      if (digraph === 'ng') { phonemes.push('NG'); i++; continue; }
    }
    
    if (LETTER_TO_PHONEME[char]) {
      phonemes.push(LETTER_TO_PHONEME[char]);
    }
  }

  return phonemes.length > 0 ? phonemes : ['SIL'];
}

/**
 * Try to run Montreal Forced Aligner
 */
async function tryMFAAlignment(
  audioPath: string,
  transcript: string
): Promise<AlignmentResult | null> {
  // Check if MFA is available
  try {
    const mfaCheck = spawn('mfa', ['version']);
    await new Promise<void>((resolve, reject) => {
      mfaCheck.on('close', code => code === 0 ? resolve() : reject());
      mfaCheck.on('error', reject);
    });
  } catch {
    console.log('  ℹ️ MFA not available, using estimation');
    return null;
  }

  // TODO: Implement full MFA pipeline
  // For now, return null to fall back to estimation
  console.log('  ℹ️ MFA integration pending, using estimation');
  return null;
}

/**
 * Generate alignment for an audio file
 */
async function generateAlignment(
  audioPath: string,
  transcript: string,
  estimatedDuration?: number
): Promise<AlignmentResult> {
  // Try MFA first
  const mfaResult = await tryMFAAlignment(audioPath, transcript);
  if (mfaResult) {
    return mfaResult;
  }

  // Fall back to estimation
  return estimateAlignment(transcript, estimatedDuration);
}

// =============================================================================
// BLENDSHAPE TIMELINE GENERATION
// =============================================================================

/**
 * Generate blendshape timeline from phone alignments
 */
function generateBlendshapeTimeline(
  phones: PhoneAlignment[],
  duration: number,
  fps: number = FPS
): BlendshapeFrame[] {
  if (phones.length === 0) {
    return [];
  }

  const frameInterval = 1 / fps;
  const timeline: BlendshapeFrame[] = [];
  
  let currentPhoneIndex = 0;
  
  for (let time = 0; time <= duration; time += frameInterval) {
    // Find current phone
    while (
      currentPhoneIndex < phones.length - 1 &&
      time >= phones[currentPhoneIndex + 1].start
    ) {
      currentPhoneIndex++;
    }
    
    const currentPhone = phones[currentPhoneIndex];
    const nextPhone = phones[currentPhoneIndex + 1];
    
    let blendshapes: Record<string, number>;
    
    if (nextPhone && time >= currentPhone.start && time < nextPhone.start) {
      // Interpolate between current and next phone
      const phoneDuration = nextPhone.start - currentPhone.start;
      const elapsed = time - currentPhone.start;
      const t = easeInOutQuad(elapsed / phoneDuration);
      
      const currentBS = VISEME_TO_BLENDSHAPES[currentPhone.viseme] || VISEME_TO_BLENDSHAPES['REST'];
      const nextBS = VISEME_TO_BLENDSHAPES[nextPhone.viseme] || VISEME_TO_BLENDSHAPES['REST'];
      
      blendshapes = interpolateBlendshapes(currentBS, nextBS, t);
    } else {
      blendshapes = { ...(VISEME_TO_BLENDSHAPES[currentPhone.viseme] || VISEME_TO_BLENDSHAPES['REST']) };
    }
    
    timeline.push({
      timestamp: Math.round(time * 1000) / 1000,
      blendshapes,
    });
  }
  
  return timeline;
}

function interpolateBlendshapes(
  from: Record<string, number>,
  to: Record<string, number>,
  t: number
): Record<string, number> {
  const result: Record<string, number> = {};
  const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
  
  for (const key of allKeys) {
    const fromValue = from[key] || 0;
    const toValue = to[key] || 0;
    result[key] = Math.round(fromValue + (toValue - fromValue) * t);
  }
  
  return result;
}

function easeInOutQuad(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// =============================================================================
// DATABASE FUNCTIONS
// =============================================================================

function createSupabaseClient() {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    throw new Error('Supabase credentials not configured');
  }
  return createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
}

async function storeAlignment(
  supabase: ReturnType<typeof createClient>,
  entry: ManifestEntry,
  alignment: AlignmentResult
): Promise<void> {
  const { error } = await supabase
    .from('lipsync_alignments')
    .upsert({
      day_number: entry.day,
      age_bucket: entry.ageBucket,
      language: entry.language,
      phase: entry.phase,
      shard_id: entry.shardId,
      transcript: entry.transcript,
      words: alignment.words,
      phones: alignment.phones,
      blendshape_timeline: alignment.blendshapeTimeline,
      duration_seconds: alignment.duration,
      method: alignment.method,
      confidence: alignment.confidence,
      fps: FPS,
    }, {
      onConflict: 'day_number,age_bucket,language,phase',
    });

  if (error) {
    console.error(`Error storing alignment:`, error);
    throw error;
  }
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

async function processManifest(manifestPath: string): Promise<void> {
  const supabase = createSupabaseClient();
  
  // Read manifest
  const manifest: ManifestEntry[] = JSON.parse(
    fs.readFileSync(manifestPath, 'utf-8')
  );
  
  console.log('\n🎯 KELLY LIPSYNC ALIGNMENT GENERATION');
  console.log('=' .repeat(50));
  console.log(`Processing ${manifest.length} entries from manifest`);
  console.log(`FPS: ${FPS}`);
  console.log('=' .repeat(50));
  
  // Ensure output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  
  let processed = 0;
  let errors = 0;
  
  for (const entry of manifest) {
    try {
      console.log(`\n📍 Day ${entry.day} | ${entry.ageBucket} | ${entry.phase}:`);
      
      // Generate alignment
      const alignment = await generateAlignment(
        entry.audioPath,
        entry.transcript,
        entry.duration
      );
      
      // Generate blendshape timeline
      alignment.blendshapeTimeline = generateBlendshapeTimeline(
        alignment.phones,
        alignment.duration,
        FPS
      );
      
      console.log(`   Words: ${alignment.words.length}`);
      console.log(`   Phones: ${alignment.phones.length}`);
      console.log(`   Frames: ${alignment.blendshapeTimeline.length}`);
      console.log(`   Method: ${alignment.method}`);
      
      // Store in Supabase
      await storeAlignment(supabase, entry, alignment);
      console.log(`   ✓ Stored in database`);
      
      // Also save local copy
      const localPath = path.join(
        OUTPUT_DIR,
        `day-${entry.day}`,
        `${entry.day}_${entry.ageBucket}_${entry.language}_${entry.phase}_alignment.json`
      );
      const localDir = path.dirname(localPath);
      if (!fs.existsSync(localDir)) {
        fs.mkdirSync(localDir, { recursive: true });
      }
      fs.writeFileSync(localPath, JSON.stringify(alignment, null, 2));
      
      processed++;
      
    } catch (error) {
      console.error(`   ❌ Error processing:`, error);
      errors++;
    }
  }
  
  console.log('\n' + '=' .repeat(50));
  console.log(`✅ ALIGNMENT COMPLETE`);
  console.log(`   Processed: ${processed}`);
  console.log(`   Errors: ${errors}`);
  console.log(`   Success rate: ${((processed / (processed + errors)) * 100).toFixed(1)}%`);
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  let manifestPath = path.join(INPUT_DIR, 'manifest.json');
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' || args[i] === '-i') {
      const inputDir = args[++i];
      manifestPath = path.join(inputDir, 'manifest.json');
    }
    if (args[i] === '--help' || args[i] === '-h') {
      console.log(`
Kelly Lipsync Alignment Generation

Usage:
  npx ts-node generate-alignments.ts [options]

Options:
  --input, -i <dir>   Input directory containing manifest.json (default: ./generated-audio)
  --help, -h          Show this help

The manifest.json should be created by generate-lesson-audio.ts
      `);
      process.exit(0);
    }
  }
  
  if (!fs.existsSync(manifestPath)) {
    console.error(`❌ Manifest not found: ${manifestPath}`);
    console.log('Run generate-lesson-audio.ts first to create audio files and manifest.');
    process.exit(1);
  }
  
  await processManifest(manifestPath);
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

