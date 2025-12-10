/**
 * 🏆 GOLDEN LESSON ALIGNMENT GENERATOR
 * 
 * Generates pre-computed phoneme alignments for the Golden Lesson (Day 1).
 * These alignments enable smooth blendshape-based lipsync in Unity when 
 * full video isn't available or for real-time fallback.
 * 
 * Output:
 * - Word-level timing
 * - Phoneme-level timing  
 * - Pre-computed blendshape timelines (30 FPS)
 * - Stored in Supabase lipsync_alignments table
 * 
 * Usage:
 *   npx tsx scripts/golden-lesson-alignment-generator.ts
 *   npx tsx scripts/golden-lesson-alignment-generator.ts --archetype "The Explorer"
 */

import 'dotenv/config';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL!,
  SUPABASE_SERVICE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-alignments', 'golden-lesson'),
  FPS: 30,
};

// =============================================================================
// PHONEME TO VISEME MAPPING (ARKit Compatible)
// =============================================================================

const PHONEME_TO_VISEME: Record<string, string> = {
  // Vowels
  'AA': 'ah', 'AE': 'ah', 'AH': 'ah', 'AO': 'oh', 'AW': 'oh',
  'AY': 'ah', 'EH': 'eh', 'ER': 'er', 'EY': 'eh', 'IH': 'ih',
  'IY': 'ee', 'OW': 'oh', 'OY': 'oh', 'UH': 'oo', 'UW': 'oo',
  
  // Consonants  
  'B': 'mb', 'CH': 'ch', 'D': 'dd', 'DH': 'th', 'F': 'ff',
  'G': 'kk', 'HH': 'hh', 'JH': 'ch', 'K': 'kk', 'L': 'nn',
  'M': 'mb', 'N': 'nn', 'NG': 'nn', 'P': 'mb', 'R': 'rr',
  'S': 'ss', 'SH': 'ch', 'T': 'dd', 'TH': 'th', 'V': 'ff',
  'W': 'oo', 'Y': 'ee', 'Z': 'ss', 'ZH': 'ch',
  
  // Silence
  'SIL': 'sil', 'SP': 'sil',
};

// =============================================================================
// VISEME TO BLENDSHAPE MAPPING
// =============================================================================

const VISEME_TO_BLENDSHAPES: Record<string, Record<string, number>> = {
  'sil': { jawOpen: 0, mouthClose: 15, mouthSmileLeft: 10, mouthSmileRight: 10 },
  'ah': { jawOpen: 50, mouthOpen: 40, mouthStretchLeft: 10, mouthStretchRight: 10 },
  'oh': { jawOpen: 40, mouthFunnel: 50, mouthPucker: 30 },
  'eh': { jawOpen: 30, mouthOpen: 25, mouthStretchLeft: 15, mouthStretchRight: 15 },
  'ee': { jawOpen: 20, mouthStretchLeft: 30, mouthStretchRight: 30, mouthSmileLeft: 20, mouthSmileRight: 20 },
  'ih': { jawOpen: 25, mouthStretchLeft: 20, mouthStretchRight: 20 },
  'oo': { jawOpen: 25, mouthFunnel: 60, mouthPucker: 50 },
  'er': { jawOpen: 20, mouthFunnel: 20 },
  'mb': { jawOpen: 5, mouthClose: 80, mouthPressLeft: 40, mouthPressRight: 40 },
  'ff': { jawOpen: 15, mouthUpperUpLeft: 20, mouthUpperUpRight: 20, mouthLowerDownLeft: 10, mouthLowerDownRight: 10 },
  'th': { jawOpen: 20, mouthOpen: 15 },
  'dd': { jawOpen: 25, mouthOpen: 20 },
  'kk': { jawOpen: 30, mouthOpen: 25 },
  'nn': { jawOpen: 20, mouthOpen: 15 },
  'ss': { jawOpen: 15, mouthStretchLeft: 20, mouthStretchRight: 20 },
  'ch': { jawOpen: 25, mouthFunnel: 30 },
  'rr': { jawOpen: 20, mouthFunnel: 20 },
  'hh': { jawOpen: 30, mouthOpen: 30 },
};

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

let supabase: SupabaseClient;

function getSupabase(): SupabaseClient {
  if (!supabase) {
    supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_SERVICE_KEY);
  }
  return supabase;
}

// =============================================================================
// TYPES
// =============================================================================

interface WordTiming {
  word: string;
  start: number;
  end: number;
}

interface PhoneTiming {
  phone: string;
  start: number;
  end: number;
  viseme: string;
}

interface BlendshapeFrame {
  timestamp: number;
  blendshapes: Record<string, number>;
}

interface AlignmentData {
  words: WordTiming[];
  phones: PhoneTiming[];
  blendshapeTimeline: BlendshapeFrame[];
  duration: number;
  method: string;
  confidence: number;
}

interface GoldenLessonAtom {
  id: string;
  archetype: string;
  phase: string;
  content: {
    script: string;
  };
}

// =============================================================================
// FETCH GOLDEN LESSON ATOMS
// =============================================================================

async function fetchGoldenLessonAtoms(archetype?: string): Promise<GoldenLessonAtom[]> {
  const sb = getSupabase();
  
  const { data: lesson } = await sb
    .from('core_lessons')
    .select('id')
    .eq('day_number', 1)
    .single();
    
  if (!lesson) throw new Error('Day 1 lesson not found');
  
  let query = sb
    .from('lesson_atoms')
    .select('id, phase, archetype, content')
    .eq('core_lesson_id', lesson.id);
  
  if (archetype) {
    query = query.eq('archetype', archetype);
  }
  
  const { data: atoms } = await query;
  return (atoms || []) as GoldenLessonAtom[];
}

// =============================================================================
// AUDIO GENERATION
// =============================================================================

async function generateAudio(text: string, outputPath: string): Promise<string> {
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
        text,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.85,
        },
      }),
    }
  );
  
  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buffer);
  
  return outputPath;
}

// =============================================================================
// ALIGNMENT ESTIMATION (Fallback when MFA not available)
// =============================================================================

function estimateAlignment(text: string, durationSeconds: number): AlignmentData {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  
  const wordTimings: WordTiming[] = [];
  const phoneTimings: PhoneTiming[] = [];
  
  let currentTime = 0.1; // Small initial pause
  
  for (const word of words) {
    // Estimate word duration proportional to length
    const wordDuration = (word.length / totalChars) * (durationSeconds - 0.2) * 0.85;
    const wordStart = currentTime;
    const wordEnd = currentTime + wordDuration;
    
    wordTimings.push({ word, start: wordStart, end: wordEnd });
    
    // Estimate phonemes (simplified: 1 phoneme per character)
    const phonesPerWord = Math.max(2, Math.floor(word.length * 0.8));
    const phoneDuration = wordDuration / phonesPerWord;
    
    for (let i = 0; i < phonesPerWord; i++) {
      const phoneStart = wordStart + (i * phoneDuration);
      const phoneEnd = phoneStart + phoneDuration;
      
      // Simple vowel/consonant estimation
      const charIndex = Math.floor((i / phonesPerWord) * word.length);
      const char = word[charIndex]?.toLowerCase() || 'a';
      
      let phone = 'AH'; // Default
      if ('aeiou'.includes(char)) {
        phone = { 'a': 'AH', 'e': 'EH', 'i': 'IH', 'o': 'OW', 'u': 'UW' }[char] || 'AH';
      } else if ('bmp'.includes(char)) {
        phone = 'M';
      } else if ('fv'.includes(char)) {
        phone = 'F';
      } else if ('sz'.includes(char)) {
        phone = 'S';
      } else if ('td'.includes(char)) {
        phone = 'T';
      } else if ('kg'.includes(char)) {
        phone = 'K';
      } else if ('lr'.includes(char)) {
        phone = char === 'l' ? 'L' : 'R';
      } else if ('n'.includes(char)) {
        phone = 'N';
      }
      
      const viseme = PHONEME_TO_VISEME[phone] || 'sil';
      phoneTimings.push({ phone, start: phoneStart, end: phoneEnd, viseme });
    }
    
    // Add small pause between words
    currentTime = wordEnd + (durationSeconds * 0.02);
  }
  
  // Generate blendshape timeline
  const blendshapeTimeline = generateBlendshapeTimeline(phoneTimings, durationSeconds);
  
  return {
    words: wordTimings,
    phones: phoneTimings,
    blendshapeTimeline,
    duration: durationSeconds,
    method: 'estimation',
    confidence: 0.7,
  };
}

// =============================================================================
// BLENDSHAPE TIMELINE GENERATION
// =============================================================================

function generateBlendshapeTimeline(
  phones: PhoneTiming[],
  duration: number
): BlendshapeFrame[] {
  const frames: BlendshapeFrame[] = [];
  const totalFrames = Math.ceil(duration * CONFIG.FPS);
  
  const restingFace = VISEME_TO_BLENDSHAPES['sil'];
  
  for (let frame = 0; frame < totalFrames; frame++) {
    const timestamp = frame / CONFIG.FPS;
    
    // Find active phoneme at this timestamp
    const activePhone = phones.find(p => timestamp >= p.start && timestamp < p.end);
    
    let targetBlendshapes: Record<string, number>;
    
    if (activePhone) {
      targetBlendshapes = VISEME_TO_BLENDSHAPES[activePhone.viseme] || restingFace;
    } else {
      targetBlendshapes = restingFace;
    }
    
    // Apply smoothing by blending with previous frame
    if (frames.length > 0) {
      const prevFrame = frames[frames.length - 1];
      const smoothedBlendshapes: Record<string, number> = {};
      const smoothing = 0.3;
      
      const allKeys = new Set([
        ...Object.keys(prevFrame.blendshapes),
        ...Object.keys(targetBlendshapes)
      ]);
      
      for (const key of allKeys) {
        const prev = prevFrame.blendshapes[key] || 0;
        const target = targetBlendshapes[key] || 0;
        smoothedBlendshapes[key] = prev + (target - prev) * (1 - smoothing);
      }
      
      frames.push({ timestamp, blendshapes: smoothedBlendshapes });
    } else {
      frames.push({ timestamp, blendshapes: { ...targetBlendshapes } });
    }
  }
  
  return frames;
}

// =============================================================================
// STORE ALIGNMENT IN SUPABASE
// =============================================================================

async function storeAlignment(
  dayNumber: number,
  phase: string,
  archetype: string,
  transcript: string,
  alignment: AlignmentData
): Promise<void> {
  const sb = getSupabase();
  
  // Map archetype to age bucket for compatibility
  const ageBucket = archetype === 'The Explorer' ? '6-12' :
                    archetype === 'The Rebel' ? '13-17' :
                    archetype === 'The Scientist' ? '18-35' : '18-35';
  
  const record = {
    day_number: dayNumber,
    age_bucket: ageBucket,
    language: 'en',
    phase: phase,
    transcript: transcript,
    words: alignment.words,
    phones: alignment.phones,
    blendshape_timeline: alignment.blendshapeTimeline,
    duration_seconds: alignment.duration,
    method: alignment.method,
    confidence: alignment.confidence,
    fps: CONFIG.FPS,
    updated_at: new Date().toISOString(),
  };
  
  const { error } = await sb
    .from('lipsync_alignments')
    .upsert(record, {
      onConflict: 'day_number,age_bucket,language,phase',
      ignoreDuplicates: false,
    });
  
  if (error) {
    console.warn(`   ⚠️ Database warning: ${error.message}`);
  }
}

// =============================================================================
// GET AUDIO DURATION (Estimation)
// =============================================================================

function estimateAudioDuration(text: string): number {
  // Average speaking rate: ~150 words per minute = 2.5 words per second
  // Average word length: ~5 characters
  const words = text.split(/\s+/).length;
  return Math.max(1, words / 2.5);
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

async function generateGoldenLessonAlignments(archetype?: string): Promise<void> {
  console.log('\n');
  console.log('╔' + '═'.repeat(68) + '╗');
  console.log('║  🏆 GOLDEN LESSON ALIGNMENT GENERATOR                               ║');
  console.log('║  Pre-computing phoneme alignments for Day 1                         ║');
  console.log('╚' + '═'.repeat(68) + '╝');
  
  // Validate keys
  if (!CONFIG.ELEVENLABS_API_KEY || !CONFIG.SUPABASE_URL) {
    console.error('❌ Missing required API keys');
    process.exit(1);
  }
  
  // Fetch atoms
  console.log('\n📊 Fetching Golden Lesson atoms...');
  const atoms = await fetchGoldenLessonAtoms(archetype);
  console.log(`   Found ${atoms.length} atoms`);
  
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  
  let successCount = 0;
  
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    console.log(`\n[${i + 1}/${atoms.length}] ${atom.archetype} - ${atom.phase}`);
    
    try {
      const script = atom.content.script;
      
      // Generate audio
      const audioPath = path.join(
        CONFIG.OUTPUT_DIR,
        `day_001_${atom.phase}_${atom.archetype.replace(/\s+/g, '_')}.mp3`
      );
      console.log(`   🎤 Generating audio...`);
      await generateAudio(script, audioPath);
      
      // Estimate duration (in production, use actual audio analysis)
      const duration = estimateAudioDuration(script);
      
      // Generate alignment (estimation fallback)
      console.log(`   📊 Generating alignment...`);
      const alignment = estimateAlignment(script, duration);
      
      // Store in database
      console.log(`   💾 Storing in Supabase...`);
      await storeAlignment(1, atom.phase, atom.archetype, script, alignment);
      
      // Save alignment file locally
      const alignmentPath = path.join(
        CONFIG.OUTPUT_DIR,
        `day_001_${atom.phase}_${atom.archetype.replace(/\s+/g, '_')}_alignment.json`
      );
      fs.writeFileSync(alignmentPath, JSON.stringify(alignment, null, 2));
      
      console.log(`   ✅ Complete (${alignment.words.length} words, ${alignment.phones.length} phones)`);
      successCount++;
      
    } catch (error: any) {
      console.log(`   ❌ Failed: ${error.message}`);
    }
    
    // Small delay
    await new Promise(r => setTimeout(r, 1000));
  }
  
  console.log('\n' + '═'.repeat(70));
  console.log(`✅ ALIGNMENT GENERATION COMPLETE: ${successCount}/${atoms.length}`);
  console.log(`📁 Files saved to: ${CONFIG.OUTPUT_DIR}`);
  console.log('═'.repeat(70));
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  let archetype: string | undefined;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--archetype') {
      archetype = args[++i];
    }
    if (args[i] === '--help') {
      console.log(`
Golden Lesson Alignment Generator

Usage:
  npx tsx scripts/golden-lesson-alignment-generator.ts [options]

Options:
  --archetype <name>   Generate only for specific archetype
  --help               Show this help
      `);
      process.exit(0);
    }
  }
  
  await generateGoldenLessonAlignments(archetype);
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

export {
  generateGoldenLessonAlignments,
  estimateAlignment,
  generateBlendshapeTimeline,
  PHONEME_TO_VISEME,
  VISEME_TO_BLENDSHAPES,
};



