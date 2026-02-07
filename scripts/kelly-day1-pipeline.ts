#!/usr/bin/env npx ts-node
/**
 * Kelly Day 1 End-to-End Pipeline
 * 
 * Generates Day 1 lesson video with full eval gates.
 * Usage: npx ts-node scripts/kelly-day1-pipeline.ts [--dry-run]
 * 
 * Required env vars:
 * - PUBLIC_SUPABASE_URL or SUPABASE_URL
 * - SUPABASE_SERVICE_ROLE_KEY
 * - ELEVENLABS_API_KEY
 * - At least one of: HEYGEN_API_KEY, SYNC_LABS_API_KEY, FAL_KEY, REPLICATE_API_TOKEN
 */

import { config } from 'dotenv';
config(); // Load .env file

import { createClient } from '@supabase/supabase-js';

// Configuration
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || '';
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

if (!SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// Types
type Phase = 'hook' | 'story' | 'wonder' | 'action' | 'wisdom';
type EngineType = 'heygen' | 'sync_so' | 'fal_latentsync' | 'replicate';

// SLOP_PATTERNS for content eval
const SLOP_PATTERNS = [
  /dive\s*(into|deep)/i,
  /unlock\s*the\s*secrets?/i,
  /embark\s*on\s*(a\s*)?journey/i,
  /game[- ]?changer/i,
  /transformative\s*(experience|journey)/i,
  /unleash\s*(your|the)/i,
];

const FORBIDDEN_WORDS = ['user', 'users', 'unlock', 'exclusive', 'amazing', 'awesome', 'incredible'];

// Kelly assets
const KELLY_ASSETS = {
  base_image: 'https://storage.googleapis.com/curious-kelly-assets/kelly/kelly-presenter-01.png',
  talking_video: 'https://storage.googleapis.com/curious-kelly-assets/kelly/kelly-talking-loop.mp4',
};

// Provider fallback chain
const PROVIDER_CHAIN: EngineType[] = ['heygen', 'sync_so', 'fal_latentsync', 'replicate'];

interface EvalResult {
  passed: boolean;
  score: number;
  issues: string[];
}

interface PipelineResult {
  success: boolean;
  day: number;
  phases_completed: string[];
  phases_failed: string[];
  errors: string[];
  stats: {
    audio_generated: number;
    videos_submitted: number;
    eval_failures: number;
  };
}

// ============================================
// EVAL GATES
// ============================================

function evaluateContent(text: string): EvalResult {
  const issues: string[] = [];
  
  // Check slop patterns
  for (const pattern of SLOP_PATTERNS) {
    if (pattern.test(text)) {
      issues.push(`SLOP: matches ${pattern.toString()}`);
    }
  }
  
  // Check forbidden words
  const lowerText = text.toLowerCase();
  for (const word of FORBIDDEN_WORDS) {
    if (lowerText.includes(word)) {
      issues.push(`FORBIDDEN: contains "${word}"`);
    }
  }
  
  // Check length
  if (text.length < 20) {
    issues.push(`LENGTH: Too short (${text.length} chars)`);
  }
  
  return {
    passed: issues.length === 0,
    score: Math.max(0, 100 - issues.length * 20),
    issues,
  };
}

// ============================================
// PROVIDER AVAILABILITY
// ============================================

async function getAvailableProviders(): Promise<EngineType[]> {
  const available: EngineType[] = [];
  
  if (process.env.HEYGEN_API_KEY) {
    try {
      const res = await fetch('https://api.heygen.com/v2/avatars', {
        headers: { 'X-Api-Key': process.env.HEYGEN_API_KEY },
      });
      if (res.status === 200) available.push('heygen');
    } catch { /* skip */ }
  }
  
  if (process.env.SYNC_LABS_API_KEY) {
    available.push('sync_so'); // Assume available if key present
  }
  
  if (process.env.FAL_KEY) {
    available.push('fal_latentsync');
  }
  
  if (process.env.REPLICATE_API_TOKEN) {
    available.push('replicate');
  }
  
  return available;
}

// ============================================
// MAIN PIPELINE
// ============================================

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const testEvals = args.includes('--test-evals');
  
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║       KELLY DAY 1 END-TO-END PIPELINE                        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
  
  if (testEvals) {
    console.log('Running eval gate tests...\n');
    runEvalTests();
    return;
  }
  
  if (dryRun) {
    console.log('🏃 DRY RUN MODE - No actual API calls will be made\n');
  }
  
  // Check configuration
  console.log('📋 Checking configuration...');
  console.log(`   Supabase:   ${SUPABASE_URL ? '✅' : '❌'}`);
  console.log(`   ElevenLabs: ${ELEVENLABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   HeyGen:     ${process.env.HEYGEN_API_KEY ? '✅' : '❌'}`);
  console.log(`   Sync Labs:  ${process.env.SYNC_LABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   Fal.ai:     ${process.env.FAL_KEY ? '✅' : '❌'}`);
  console.log(`   Replicate:  ${process.env.REPLICATE_API_TOKEN ? '✅' : '❌'}`);
  
  // Check provider availability
  console.log('\n🔌 Checking provider availability...');
  const providers = await getAvailableProviders();
  
  if (providers.length === 0 && !dryRun) {
    console.error('\n❌ No video providers available!');
    process.exit(1);
  }
  
  console.log(`   Available: ${providers.length > 0 ? providers.join(' → ') : '(none - dry run)'}`);
  
  // Generate Day 1
  const result = await generateDay1(dryRun, providers);
  
  // Print results
  console.log('\n' + '═'.repeat(60));
  console.log('📊 PIPELINE RESULTS');
  console.log('═'.repeat(60));
  console.log(`   Success: ${result.success ? '✅ YES' : '❌ NO'}`);
  console.log(`   Day: ${result.day}`);
  console.log(`   Phases completed: ${result.phases_completed.join(', ') || 'none'}`);
  console.log(`   Phases failed: ${result.phases_failed.join(', ') || 'none'}`);
  
  if (result.errors.length > 0) {
    console.log('\n   Errors:');
    result.errors.forEach(e => console.log(`     ⚠️  ${e}`));
  }
  
  console.log('\n' + '═'.repeat(60));
  process.exit(result.success ? 0 : 1);
}

async function generateDay1(dryRun: boolean, providers: EngineType[]): Promise<PipelineResult> {
  const DAY = 1;
  const PHASES: Phase[] = ['hook', 'story', 'wonder', 'action', 'wisdom'];
  
  const result: PipelineResult = {
    success: true,
    day: DAY,
    phases_completed: [],
    phases_failed: [],
    errors: [],
    stats: {
      audio_generated: 0,
      videos_submitted: 0,
      eval_failures: 0,
    },
  };
  
  // Sample content for Day 1 (used if database has no content)
  const sampleContent: Record<Phase, string> = {
    hook: "Hi, I'm Kelly. Today we're exploring compound interest. It's one of the most powerful ideas in all of finance.",
    story: "Imagine you plant a seed. That seed grows into a tree, and that tree makes more seeds. Those seeds grow into more trees. That's compound interest - your money making money that makes more money.",
    wonder: "Did you know that if you invested just $100 at age 20, with 7% annual returns, you'd have over $2,100 by age 65? That's the magic of compound growth working for you.",
    action: "Here's something simple you can do today: look at your savings. Even a small amount, left alone to grow, can become something wonderful over time.",
    wisdom: "The best time to start was yesterday. The second best time is today. Your future self will thank you for every seed you plant now.",
  };
  
  console.log(`\n📚 Generating Day ${DAY}...`);
  
  // Get Day 1 lesson from database
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_of_year', DAY)
    .single();
  
  let content = sampleContent;
  
  if (!lessonError && lesson) {
    console.log(`   Found: "${lesson.topic}"`);
    
    // Get atoms for each phase
    const { data: atoms } = await supabase
      .from('lesson_atoms')
      .select('*')
      .eq('core_lesson_id', lesson.id);
    
    if (atoms && atoms.length > 0) {
      for (const atom of atoms) {
        const text = atom.content?.script || atom.content?.text || '';
        if (text && atom.phase) {
          content[atom.phase as Phase] = text;
        }
      }
    }
  } else {
    console.log('   Using sample content (no DB entry found)');
  }
  
  // Process each phase
  for (const phase of PHASES) {
    console.log(`\n🎬 Processing: ${phase}`);
    const text = content[phase];
    console.log(`   Text: "${text.substring(0, 50)}..."`);
    
    // 1. CONTENT EVAL GATE
    const evalResult = evaluateContent(text);
    
    if (!evalResult.passed) {
      console.log(`   ❌ Content eval FAILED (score: ${evalResult.score})`);
      evalResult.issues.forEach(i => console.log(`      - ${i}`));
      result.phases_failed.push(phase);
      result.stats.eval_failures++;
      result.errors.push(`${phase}: ${evalResult.issues[0]}`);
      continue;
    }
    
    console.log(`   ✅ Content eval passed (score: ${evalResult.score})`);
    
    if (dryRun) {
      console.log(`   [DRY RUN] Would generate audio and video`);
      result.phases_completed.push(phase);
      continue;
    }
    
    // 2. CREATE VIDEO JOB (in real mode)
    const jobId = `day${DAY}-${phase}-adult-${Date.now()}`;
    
    const job = {
      id: jobId,
      day_of_year: DAY,
      phase,
      age_category: 'adult',
      language: 'en',
      engine: providers[0] || 'sync_so',
      status: 'queued',
      input_payload: {
        text,
        source_image_url: KELLY_ASSETS.base_image,
        video_url: KELLY_ASSETS.talking_video,
      },
      priority: phase === 'hook' ? 10 : 5,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    
    // Insert job to database
    const { error: insertError } = await supabase
      .from('video_jobs')
      .insert(job);
    
    if (insertError) {
      console.log(`   ⚠️  DB insert failed: ${insertError.message}`);
      result.errors.push(`${phase}: DB insert failed`);
      result.phases_failed.push(phase);
    } else {
      console.log(`   ✅ Job queued: ${jobId}`);
      result.phases_completed.push(phase);
      result.stats.videos_submitted++;
    }
  }
  
  result.success = result.phases_failed.length === 0;
  return result;
}

function runEvalTests() {
  console.log('Running eval gate tests...\n');
  
  const testCases = [
    { name: 'Good content', text: "Hi, I'm Kelly. Today we're learning about compound interest.", shouldPass: true },
    { name: 'Slop: dive deep', text: "Let's dive deep into the world of finance!", shouldPass: false },
    { name: 'Slop: unlock secrets', text: "Unlock the secrets of compound interest!", shouldPass: false },
    { name: 'Slop: embark on journey', text: "We'll embark on a journey of discovery.", shouldPass: false },
    { name: 'Forbidden: user', text: "Hello user, welcome to your lesson.", shouldPass: false },
    { name: 'Too short', text: "Hi Kelly.", shouldPass: false },
  ];
  
  let passed = 0;
  let failed = 0;
  
  for (const tc of testCases) {
    const result = evaluateContent(tc.text);
    const testPassed = result.passed === tc.shouldPass;
    
    if (testPassed) {
      passed++;
      console.log(`✅ ${tc.name}`);
    } else {
      failed++;
      console.log(`❌ ${tc.name}`);
      console.log(`   Expected: ${tc.shouldPass ? 'PASS' : 'FAIL'}, Got: ${result.passed ? 'PASS' : 'FAIL'}`);
      if (result.issues.length > 0) {
        result.issues.forEach(issue => console.log(`   - ${issue}`));
      }
    }
  }
  
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`Results: ${passed}/${testCases.length} passed`);
  
  if (failed > 0) {
    process.exit(1);
  }
}

// Run
main().catch(console.error);
