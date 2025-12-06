/**
 * Kelly Lipsync Pipeline - Audio Generation
 * 
 * Generates audio files for lessons using ElevenLabs API.
 * Creates one audio file per day × age bucket × language combination.
 * 
 * Usage:
 *   npx ts-node scripts/lipsync-pipeline/generate-lesson-audio.ts --days 1-30 --ages all --lang en
 *   npx ts-node scripts/lipsync-pipeline/generate-lesson-audio.ts --day 1 --age 6-12 --lang en
 */

import { createClient } from '@supabase/supabase-js';
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

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || process.env.ELEVENLABS_VOICE_ID || 'kelly-voice-id'; // Kelly's trained voice

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const OUTPUT_DIR = path.join(process.cwd(), 'generated-audio');

// Age bucket mapping
const AGE_BUCKETS: Record<string, { min: number; max: number }> = {
  '2-5': { min: 2, max: 5 },
  '6-12': { min: 6, max: 12 },
  '13-17': { min: 13, max: 17 },
  '18-35': { min: 18, max: 35 },
  '36-60': { min: 36, max: 60 },
  '61+': { min: 61, max: 102 },
};

// Rate limiting
const REQUESTS_PER_MINUTE = 10;
const REQUEST_INTERVAL = 60000 / REQUESTS_PER_MINUTE;

// =============================================================================
// TYPES
// =============================================================================

interface LessonShard {
  id: string;
  age: number;
  region: string;
  tone: string;
  script_content: {
    script: string;
    options?: string[];
    responses?: Record<string, string>;
  };
  day_number: number;
  topic: string;
}

interface AudioGenerationResult {
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
// ELEVENLABS API
// =============================================================================

async function generateAudioElevenLabs(
  text: string,
  outputPath: string
): Promise<{ duration: number }> {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY not set');
  }

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.5,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`ElevenLabs API error: ${response.status} - ${error}`);
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  
  // Ensure directory exists
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  
  fs.writeFileSync(outputPath, audioBuffer);
  
  // Estimate duration (rough: ~150 words per minute)
  const wordCount = text.split(/\s+/).length;
  const estimatedDuration = (wordCount / 150) * 60;
  
  console.log(`✓ Generated: ${outputPath} (${text.length} chars, ~${estimatedDuration.toFixed(1)}s)`);
  
  return { duration: estimatedDuration };
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

async function getShardForAgeBucket(
  supabase: ReturnType<typeof createClient>,
  dayNumber: number,
  ageBucket: string,
  language: string
): Promise<LessonShard | null> {
  const { min, max } = AGE_BUCKETS[ageBucket];
  
  // Get a representative shard for this age bucket
  // Prefer middle of range and common tones
  const { data, error } = await supabase
    .from('lesson_shards')
    .select(`
      id,
      age,
      region,
      tone,
      script_content,
      core_lessons!inner(day_number, topic)
    `)
    .eq('region', language)
    .gte('age', min)
    .lte('age', max)
    .eq('core_lessons.day_number', dayNumber)
    .order('age', { ascending: true })
    .limit(1)
    .single();

  if (error || !data) {
    console.warn(`No shard found for day ${dayNumber}, age ${ageBucket}, lang ${language}`);
    return null;
  }

  // @ts-ignore - flatten the joined data
  return {
    ...data,
    day_number: data.core_lessons.day_number,
    topic: data.core_lessons.topic,
  };
}

async function getAllShardsForDay(
  supabase: ReturnType<typeof createClient>,
  dayNumber: number,
  language: string
): Promise<Map<string, LessonShard>> {
  const shards = new Map<string, LessonShard>();
  
  for (const ageBucket of Object.keys(AGE_BUCKETS)) {
    const shard = await getShardForAgeBucket(supabase, dayNumber, ageBucket, language);
    if (shard) {
      shards.set(ageBucket, shard);
    }
  }
  
  return shards;
}

// =============================================================================
// GENERATION PIPELINE
// =============================================================================

async function generateAudioForShard(
  shard: LessonShard,
  ageBucket: string,
  outputDir: string
): Promise<AudioGenerationResult[]> {
  const results: AudioGenerationResult[] = [];
  const dayDir = path.join(outputDir, `day-${shard.day_number}`);
  
  // Generate main script audio
  if (shard.script_content?.script) {
    const filename = `${shard.day_number}_${ageBucket}_${shard.region}_script.mp3`;
    const outputPath = path.join(dayDir, filename);
    
    try {
      const { duration } = await generateAudioElevenLabs(
        shard.script_content.script,
        outputPath
      );
      
      results.push({
        day: shard.day_number,
        ageBucket,
        language: shard.region,
        phase: 'script',
        transcript: shard.script_content.script,
        audioPath: outputPath,
        shardId: shard.id,
        duration,
      });
    } catch (error) {
      console.error(`Error generating audio for day ${shard.day_number}:`, error);
    }
  }
  
  // Generate response audio (for interactive choices)
  if (shard.script_content?.responses) {
    const responses = shard.script_content.responses;
    const options = shard.script_content.options || [];
    
    for (let i = 0; i < options.length; i++) {
      const option = options[i];
      const response = responses[option];
      
      if (response) {
        const letter = String.fromCharCode(65 + i); // A, B, C
        const filename = `${shard.day_number}_${ageBucket}_${shard.region}_response_${letter}.mp3`;
        const outputPath = path.join(dayDir, filename);
        
        try {
          // Rate limit
          await sleep(REQUEST_INTERVAL);
          
          const { duration } = await generateAudioElevenLabs(response, outputPath);
          
          results.push({
            day: shard.day_number,
            ageBucket,
            language: shard.region,
            phase: `response_${letter}`,
            transcript: response,
            audioPath: outputPath,
            shardId: shard.id,
            duration,
          });
        } catch (error) {
          console.error(`Error generating response ${letter} for day ${shard.day_number}:`, error);
        }
      }
    }
  }
  
  return results;
}

async function generateAudioBatch(
  startDay: number,
  endDay: number,
  ageBuckets: string[],
  language: string
): Promise<AudioGenerationResult[]> {
  const supabase = createSupabaseClient();
  const allResults: AudioGenerationResult[] = [];
  
  console.log('\n🎙️ KELLY LIPSYNC AUDIO GENERATION');
  console.log('=' .repeat(50));
  console.log(`Days: ${startDay}-${endDay}`);
  console.log(`Age Buckets: ${ageBuckets.join(', ')}`);
  console.log(`Language: ${language}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log('=' .repeat(50));
  
  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  
  for (let day = startDay; day <= endDay; day++) {
    console.log(`\n📅 Day ${day}:`);
    
    for (const ageBucket of ageBuckets) {
      const shard = await getShardForAgeBucket(supabase, day, ageBucket, language);
      
      if (!shard) {
        console.log(`  ⚠️ ${ageBucket}: No content found`);
        continue;
      }
      
      console.log(`  🎯 ${ageBucket}: "${shard.topic}" (${shard.tone})`);
      
      // Rate limit between shards
      await sleep(REQUEST_INTERVAL);
      
      const results = await generateAudioForShard(shard, ageBucket, OUTPUT_DIR);
      allResults.push(...results);
      
      console.log(`     ✓ Generated ${results.length} audio file(s)`);
    }
  }
  
  // Save manifest
  const manifestPath = path.join(OUTPUT_DIR, 'manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(allResults, null, 2));
  console.log(`\n📋 Manifest saved: ${manifestPath}`);
  
  return allResults;
}

// =============================================================================
// UTILITIES
// =============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function parseArgs() {
  const args = process.argv.slice(2);
  const config: {
    startDay: number;
    endDay: number;
    ageBuckets: string[];
    language: string;
  } = {
    startDay: 1,
    endDay: 30,
    ageBuckets: Object.keys(AGE_BUCKETS),
    language: 'en',
  };
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--days':
        const days = args[++i];
        if (days.includes('-')) {
          const [start, end] = days.split('-').map(Number);
          config.startDay = start;
          config.endDay = end;
        } else {
          config.startDay = config.endDay = Number(days);
        }
        break;
        
      case '--day':
        config.startDay = config.endDay = Number(args[++i]);
        break;
        
      case '--ages':
        const ages = args[++i];
        if (ages === 'all') {
          config.ageBuckets = Object.keys(AGE_BUCKETS);
        } else {
          config.ageBuckets = ages.split(',');
        }
        break;
        
      case '--age':
        config.ageBuckets = [args[++i]];
        break;
        
      case '--lang':
        config.language = args[++i];
        break;
        
      case '--help':
        console.log(`
Kelly Lipsync Audio Generation

Usage:
  npx ts-node generate-lesson-audio.ts [options]

Options:
  --days <range>    Day range (e.g., "1-30" or "1")
  --day <number>    Single day
  --ages <list>     Age buckets (e.g., "6-12,18-35" or "all")
  --age <bucket>    Single age bucket
  --lang <code>     Language code (default: "en")
  --help            Show this help

Examples:
  npx ts-node generate-lesson-audio.ts --days 1-5 --ages all --lang en
  npx ts-node generate-lesson-audio.ts --day 1 --age 6-12 --lang en
        `);
        process.exit(0);
    }
  }
  
  return config;
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  try {
    const config = parseArgs();
    
    if (!ELEVENLABS_API_KEY) {
      console.error('❌ ELEVENLABS_API_KEY not set in environment');
      console.log('Set it in .env or export ELEVENLABS_API_KEY=your_key');
      process.exit(1);
    }
    
    const results = await generateAudioBatch(
      config.startDay,
      config.endDay,
      config.ageBuckets,
      config.language
    );
    
    console.log('\n✅ GENERATION COMPLETE');
    console.log(`   Total files: ${results.length}`);
    console.log(`   Output: ${OUTPUT_DIR}`);
    
    // Summary by day
    const byDay = new Map<number, number>();
    results.forEach(r => {
      byDay.set(r.day, (byDay.get(r.day) || 0) + 1);
    });
    
    console.log('\nFiles per day:');
    byDay.forEach((count, day) => {
      console.log(`   Day ${day}: ${count} files`);
    });
    
  } catch (error) {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  }
}

main();

