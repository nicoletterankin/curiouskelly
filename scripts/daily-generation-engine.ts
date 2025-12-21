#!/usr/bin/env npx tsx
/**
 * 🚀 CURIOUS KELLY DAILY GENERATION ENGINE
 * 
 * Generates video content for upcoming days with full safety guards:
 * - Daily cost limits
 * - Detailed logging
 * - Idempotent (skips already-completed phases)
 * - Retry failed phases
 * - Human review gate (nothing auto-publishes)
 * 
 * Usage:
 *   npx tsx scripts/daily-generation-engine.ts                    # Generate next 3 days
 *   npx tsx scripts/daily-generation-engine.ts --days=354,355,356 # Specific days
 *   npx tsx scripts/daily-generation-engine.ts --dry-run          # Preview only
 *   npx tsx scripts/daily-generation-engine.ts --retry-failed     # Retry failed phases
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import { alert } from './alert.js';

// Inline config to avoid ESM import issues
const DAILY_LIMITS = {
  heygen_credits: 100,
  elevenlabs_characters: 500000,
  max_usd: 50,
  max_retries: 3,
  generation_timeout_ms: 30 * 60 * 1000,
};

const COST_ESTIMATES = {
  heygen_per_credit_usd: 0.20,
  elevenlabs_per_1k_chars_usd: 0.30,
  openai_per_1k_tokens_usd: 0.01,
};

const GENERATION_CONFIG = {
  mvp_archetype: 'The Explorer',
  mvp_age_bucket: 'adult',
  lookahead_days: 3,
  phases: ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'] as const,
  heygen_poll_interval_ms: 30000,
  video_bucket: 'kelly-videos',
};

// =============================================================================
// CONFIGURATION
// =============================================================================

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

const LOGS_DIR = path.join(process.cwd(), 'logs', 'generation');

// =============================================================================
// VALIDATION
// =============================================================================

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// =============================================================================
// TYPES
// =============================================================================

interface DayResult {
  status: 'success' | 'partial' | 'failed' | 'skipped';
  videos_generated: number;
  videos_skipped: number;
  videos_failed: number;
  heygen_cost_credits: number;
  elevenlabs_characters: number;
  duration_seconds: number;
  phases: Record<string, 'success' | 'failed' | 'skipped'>;
  errors: string[];
}

interface GenerationLog {
  date: string;
  started_at: string;
  completed_at: string;
  target_days: number[];
  archetype: string;
  age_bucket: string;
  dry_run: boolean;
  results: Record<number, DayResult>;
  total_cost_usd: number;
  total_heygen_credits: number;
  total_elevenlabs_characters: number;
  errors: string[];
  aborted: boolean;
  abort_reason?: string;
}

interface CostTracker {
  heygen_credits: number;
  elevenlabs_characters: number;
  estimated_usd: number;
}

// =============================================================================
// ARGUMENTS
// =============================================================================

function parseArgs() {
  const args = process.argv.slice(2);
  let days: number[] = [];
  let dryRun = false;
  let retryFailed = false;
  let archetype = GENERATION_CONFIG.mvp_archetype;
  let ageBucket = GENERATION_CONFIG.mvp_age_bucket;
  
  for (const arg of args) {
    if (arg.startsWith('--days=')) {
      days = arg.split('=')[1].split(',').map(d => parseInt(d.trim(), 10));
    } else if (arg === '--dry-run') {
      dryRun = true;
    } else if (arg === '--retry-failed') {
      retryFailed = true;
    } else if (arg.startsWith('--archetype=')) {
      archetype = arg.split('=')[1];
    } else if (arg.startsWith('--age=')) {
      ageBucket = arg.split('=')[1];
    }
  }
  
  // Default: next 3 days
  if (days.length === 0) {
    const today = getTodayDayNumber();
    days = [today + 1, today + 2, today + 3].filter(d => d <= 365);
  }
  
  return { days, dryRun, retryFailed, archetype, ageBucket };
}

function getTodayDayNumber(): number {
  const startDate = new Date('2025-01-01');
  const today = new Date();
  const diffTime = today.getTime() - startDate.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.min(365, Math.max(1, diffDays));
}

// =============================================================================
// COST TRACKING
// =============================================================================

async function getTodayCosts(): Promise<CostTracker> {
  const today = new Date().toISOString().split('T')[0];
  
  const { data } = await supabase
    .from('generation_costs')
    .select('credits_used, estimated_usd, service')
    .eq('cost_date', today);
  
  const costs: CostTracker = {
    heygen_credits: 0,
    elevenlabs_characters: 0,
    estimated_usd: 0,
  };
  
  for (const row of data || []) {
    if (row.service === 'heygen') {
      costs.heygen_credits += row.credits_used || 0;
    } else if (row.service === 'elevenlabs') {
      costs.elevenlabs_characters += row.credits_used || 0;
    }
    costs.estimated_usd += row.estimated_usd || 0;
  }
  
  return costs;
}

async function recordCost(
  day: number,
  phase: string,
  service: 'heygen' | 'elevenlabs' | 'openai',
  creditsUsed: number,
  estimatedUsd: number,
  archetype: string,
  ageBucket: string
) {
  const today = new Date().toISOString().split('T')[0];
  
  await supabase.from('generation_costs').insert({
    cost_date: today,
    lesson_day: day,
    phase,
    archetype,
    age_bucket: ageBucket,
    service,
    credits_used: creditsUsed,
    estimated_usd: estimatedUsd,
  });
}

function checkLimits(costs: CostTracker): { ok: boolean; reason?: string } {
  if (costs.heygen_credits >= DAILY_LIMITS.heygen_credits) {
    return { ok: false, reason: `HeyGen credits limit reached: ${costs.heygen_credits}/${DAILY_LIMITS.heygen_credits}` };
  }
  if (costs.elevenlabs_characters >= DAILY_LIMITS.elevenlabs_characters) {
    return { ok: false, reason: `ElevenLabs characters limit reached: ${costs.elevenlabs_characters}/${DAILY_LIMITS.elevenlabs_characters}` };
  }
  if (costs.estimated_usd >= DAILY_LIMITS.max_usd) {
    return { ok: false, reason: `Daily USD limit reached: $${costs.estimated_usd.toFixed(2)}/$${DAILY_LIMITS.max_usd}` };
  }
  return { ok: true };
}

// =============================================================================
// LESSON DATA
// =============================================================================

async function getLessonAtoms(day: number) {
  // First get the core_lesson_id for this day
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', day)
    .single();
  
  if (lessonError || !lesson) {
    console.log(`   ⚠️  No core_lesson found for day ${day}`);
    return [];
  }
  
  // Then get atoms for this lesson
  const { data, error } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);
  
  if (error) throw new Error(`Failed to fetch lesson_atoms: ${error.message}`);
  return data || [];
}

async function getExistingVideos(day: number, archetype: string, ageBucket: string) {
  const { data } = await supabase
    .from('kelly_video_assets')
    .select('phase, public_url, status')
    .eq('lesson_day', day)
    .eq('template', archetype)
    .eq('age_bucket', ageBucket);
  
  const existing: Record<string, boolean> = {};
  for (const v of data || []) {
    if (v.public_url && v.status !== 'failed') {
      existing[v.phase] = true;
    }
  }
  return existing;
}

async function getFailedPhases(day: number, archetype: string, ageBucket: string) {
  const { data } = await supabase
    .from('generation_status')
    .select('phases_failed')
    .eq('lesson_day', day)
    .eq('archetype', archetype)
    .eq('age_bucket', ageBucket)
    .single();
  
  return data?.phases_failed || {};
}

// =============================================================================
// GENERATION FUNCTIONS
// =============================================================================

async function generateElevenLabsAudio(script: string): Promise<{ buffer: Buffer; characters: number }> {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY not set');
  }
  
  const resp = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}?output_format=mp3_44100_192`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );
  
  if (!resp.ok) {
    const error = await resp.text();
    throw new Error(`ElevenLabs API error: ${error}`);
  }
  
  const buffer = Buffer.from(await resp.arrayBuffer());
  return { buffer, characters: script.length };
}

async function uploadAudio(buffer: Buffer, day: number, phase: string): Promise<string> {
  const fileName = `heygen/audio/day_${day}_${phase}_${Date.now()}.mp3`;
  
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(fileName, buffer, { contentType: 'audio/mpeg', upsert: true });
  
  if (error) throw new Error(`Audio upload failed: ${error.message}`);
  
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(fileName);
  return data.publicUrl;
}

async function generateHeyGenVideo(avatarId: string, audioUrl: string): Promise<string> {
  if (!HEYGEN_API_KEY) {
    throw new Error('HEYGEN_API_KEY not set');
  }
  
  const resp = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: avatarId,
        },
        voice: {
          type: 'audio',
          audio_url: audioUrl,
        },
      }],
      dimension: { width: 1920, height: 1080 },
    }),
  });
  
  if (!resp.ok) {
    const error = await resp.text();
    throw new Error(`HeyGen API error: ${error}`);
  }
  
  const result = await resp.json();
  return result.data.video_id;
}

async function waitForHeyGenVideo(videoId: string, timeoutMs: number = DAILY_LIMITS.generation_timeout_ms): Promise<string> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeoutMs) {
    const resp = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! },
    });
    
    const result = await resp.json();
    const status = result.data?.status;
    
    if (status === 'completed') {
      return result.data.video_url;
    } else if (status === 'failed') {
      throw new Error(`HeyGen video failed: ${result.data?.error || 'unknown'}`);
    }
    
    // Wait before polling again
    await new Promise(r => setTimeout(r, GENERATION_CONFIG.heygen_poll_interval_ms));
  }
  
  throw new Error('HeyGen video generation timed out');
}

async function downloadAndUploadVideo(videoUrl: string, day: number, phase: string, archetype: string): Promise<string> {
  const resp = await fetch(videoUrl);
  if (!resp.ok) throw new Error('Failed to download HeyGen video');
  
  const buffer = Buffer.from(await resp.arrayBuffer());
  const archetypeSlug = archetype.toLowerCase().replace(/^the\s+/, '').replace(/\s+/g, '_');
  const fileName = `production/day_${String(day).padStart(3, '0')}/day_${String(day).padStart(3, '0')}_${phase}_${archetypeSlug}.mp4`;
  
  const { error } = await supabase.storage
    .from('kelly-videos')
    .upload(fileName, buffer, { contentType: 'video/mp4', upsert: true });
  
  if (error) throw new Error(`Video upload failed: ${error.message}`);
  
  const { data } = supabase.storage.from('kelly-videos').getPublicUrl(fileName);
  return data.publicUrl;
}

async function registerVideoAsset(
  day: number,
  phase: string,
  archetype: string,
  ageBucket: string,
  publicUrl: string
) {
  await supabase.from('kelly_video_assets').upsert({
    lesson_day: day,
    phase,
    template: archetype,
    age_bucket: ageBucket,
    public_url: publicUrl,
    status: 'validated',
    quality_tier: 'production',
    language: 'en',
    storage_path: publicUrl,
  }, {
    onConflict: 'lesson_day,phase,template,age_bucket,language',
    ignoreDuplicates: false,
  });
}

// =============================================================================
// STATUS UPDATES
// =============================================================================

async function updateGenerationStatus(
  day: number,
  archetype: string,
  ageBucket: string,
  status: string,
  phasesCompleted: Record<string, boolean>,
  phasesFailed: Record<string, string>
) {
  const now = new Date().toISOString();
  
  await supabase.from('generation_status').upsert({
    lesson_day: day,
    archetype,
    age_bucket: ageBucket,
    status,
    phases_completed: phasesCompleted,
    phases_failed: phasesFailed,
    generated_at: now,
    video_ready: Object.keys(phasesCompleted).length >= 7,
    year: 2025,
  }, {
    onConflict: 'lesson_day,archetype,age_bucket,year',
  });
}

// =============================================================================
// AVATAR LOOKUP
// =============================================================================

function getTalkingPhotoId(archetype: string, ageBucket: string): string {
  const idMapPath = path.join(
    process.cwd(),
    'generated-images',
    'kelly-archetypes-head-only',
    'age',
    ageBucket,
    'heygen_talking_photo_ids.json'
  );
  
  if (!fs.existsSync(idMapPath)) {
    throw new Error(`Talking photo ID map not found: ${idMapPath}`);
  }
  
  const map = JSON.parse(fs.readFileSync(idMapPath, 'utf-8'));
  const persona = archetype.toLowerCase().replace(/^the\s+/, '');
  
  if (!map[persona] || map[persona] === 'PASTE_ID_HERE') {
    throw new Error(`No talking photo ID for ${archetype}/${ageBucket}`);
  }
  
  return map[persona];
}

// =============================================================================
// PHASE GENERATION
// =============================================================================

const PHASE_MAP: Record<string, string> = {
  hook: 'Hook',
  cliff: 'Cliff',
  q1: 'Fact1',
  q2: 'Fact2',
  q3: 'Fact3',
  wisdom: 'Wisdom',
  outro: 'Outro',
};

async function generatePhase(
  day: number,
  phase: string,
  script: string,
  archetype: string,
  ageBucket: string,
  costs: CostTracker,
  dryRun: boolean
): Promise<{ success: boolean; error?: string; credits?: number; characters?: number }> {
  console.log(`    📽️  Phase: ${phase}`);
  
  if (dryRun) {
    console.log(`       [DRY RUN] Would generate video for: "${script.substring(0, 50)}..."`);
    return { success: true, credits: 1, characters: script.length };
  }
  
  try {
    // Check limits before generating
    const limitCheck = checkLimits(costs);
    if (!limitCheck.ok) {
      return { success: false, error: limitCheck.reason };
    }
    
    // 1. Generate audio
    console.log(`       🔊 Generating audio (${script.length} chars)...`);
    const { buffer: audioBuffer, characters } = await generateElevenLabsAudio(script);
    const audioUrl = await uploadAudio(audioBuffer, day, phase);
    
    // Record ElevenLabs cost
    const elevenLabsCost = (characters / 1000) * COST_ESTIMATES.elevenlabs_per_1k_chars_usd;
    await recordCost(day, phase, 'elevenlabs', characters, elevenLabsCost, archetype, ageBucket);
    costs.elevenlabs_characters += characters;
    costs.estimated_usd += elevenLabsCost;
    
    // 2. Get avatar ID
    const avatarId = getTalkingPhotoId(archetype, ageBucket);
    
    // 3. Generate HeyGen video
    console.log(`       🎬 Submitting to HeyGen...`);
    const videoId = await generateHeyGenVideo(avatarId, audioUrl);
    
    // 4. Wait for completion
    console.log(`       ⏳ Waiting for HeyGen (video_id: ${videoId})...`);
    const heygenVideoUrl = await waitForHeyGenVideo(videoId);
    
    // 5. Download and upload to Supabase
    console.log(`       📤 Uploading to Supabase...`);
    const publicUrl = await downloadAndUploadVideo(heygenVideoUrl, day, phase, archetype);
    
    // 6. Register in database
    await registerVideoAsset(day, phase, archetype, ageBucket, publicUrl);
    
    // Record HeyGen cost (1 credit per video)
    const heygenCost = 1 * COST_ESTIMATES.heygen_per_credit_usd;
    await recordCost(day, phase, 'heygen', 1, heygenCost, archetype, ageBucket);
    costs.heygen_credits += 1;
    costs.estimated_usd += heygenCost;
    
    console.log(`       ✅ Done!`);
    return { success: true, credits: 1, characters };
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.log(`       ❌ Failed: ${errorMsg}`);
    return { success: false, error: errorMsg };
  }
}

// =============================================================================
// DAY GENERATION
// =============================================================================

async function generateDay(
  day: number,
  archetype: string,
  ageBucket: string,
  costs: CostTracker,
  dryRun: boolean,
  retryFailed: boolean
): Promise<DayResult> {
  const startTime = Date.now();
  const result: DayResult = {
    status: 'success',
    videos_generated: 0,
    videos_skipped: 0,
    videos_failed: 0,
    heygen_cost_credits: 0,
    elevenlabs_characters: 0,
    duration_seconds: 0,
    phases: {},
    errors: [],
  };
  
  console.log(`\n📅 Day ${day} (${archetype} / ${ageBucket})`);
  
  // Get lesson atoms
  const atoms = await getLessonAtoms(day);
  if (atoms.length === 0) {
    console.log(`   ⚠️  No lesson_atoms found for day ${day}`);
    result.status = 'skipped';
    result.errors.push('No lesson_atoms found');
    return result;
  }
  
  // Get existing videos
  const existing = await getExistingVideos(day, archetype, ageBucket);
  const failed = retryFailed ? {} : await getFailedPhases(day, archetype, ageBucket);
  
  const phasesCompleted: Record<string, boolean> = {};
  const phasesFailed: Record<string, string> = {};
  
  // Update status to in_progress
  await updateGenerationStatus(day, archetype, ageBucket, 'in_progress', phasesCompleted, phasesFailed);
  
  // Generate each phase
  for (const phase of GENERATION_CONFIG.phases) {
    // Skip if already exists
    if (existing[phase]) {
      console.log(`    ⏭️  ${phase}: already exists, skipping`);
      result.phases[phase] = 'skipped';
      result.videos_skipped++;
      phasesCompleted[phase] = true;
      continue;
    }
    
    // Skip if previously failed (unless --retry-failed)
    if (failed[phase] && !retryFailed) {
      console.log(`    ⏭️  ${phase}: previously failed, skipping (use --retry-failed)`);
      result.phases[phase] = 'skipped';
      result.videos_skipped++;
      continue;
    }
    
    // Find script for this phase
    const atomPhase = PHASE_MAP[phase] || phase;
    const atom = atoms.find(a => a.phase === atomPhase || a.phase?.toLowerCase() === phase);
    
    if (!atom?.content?.script) {
      console.log(`    ⚠️  ${phase}: no script found`);
      result.phases[phase] = 'skipped';
      result.videos_skipped++;
      continue;
    }
    
    // Check cost limits
    const limitCheck = checkLimits(costs);
    if (!limitCheck.ok) {
      console.log(`    🛑 ${limitCheck.reason}`);
      result.status = 'partial';
      result.errors.push(limitCheck.reason!);
      phasesFailed[phase] = limitCheck.reason!;
      break;
    }
    
    // Generate
    const phaseResult = await generatePhase(
      day,
      phase,
      atom.content.script,
      archetype,
      ageBucket,
      costs,
      dryRun
    );
    
    if (phaseResult.success) {
      result.phases[phase] = 'success';
      result.videos_generated++;
      result.heygen_cost_credits += phaseResult.credits || 0;
      result.elevenlabs_characters += phaseResult.characters || 0;
      phasesCompleted[phase] = true;
    } else {
      result.phases[phase] = 'failed';
      result.videos_failed++;
      result.errors.push(`${phase}: ${phaseResult.error}`);
      phasesFailed[phase] = phaseResult.error || 'Unknown error';
    }
  }
  
  result.duration_seconds = Math.round((Date.now() - startTime) / 1000);
  
  // Determine final status
  if (result.videos_failed > 0 && result.videos_generated === 0) {
    result.status = 'failed';
  } else if (result.videos_failed > 0) {
    result.status = 'partial';
  } else if (result.videos_generated === 0 && result.videos_skipped === GENERATION_CONFIG.phases.length) {
    result.status = 'skipped';
  }
  
  // Update status
  const finalStatus = result.status === 'success' || result.status === 'skipped' 
    ? 'pending_review' 
    : result.status;
  await updateGenerationStatus(day, archetype, ageBucket, finalStatus, phasesCompleted, phasesFailed);
  
  // Alert if needed
  if (result.status === 'success') {
    await alert('REVIEW_NEEDED', { day });
  } else if (result.status === 'failed') {
    await alert('GENERATION_FAILED', { day, error: result.errors.join('; ') });
  }
  
  return result;
}

// =============================================================================
// LOGGING
// =============================================================================

function writeLog(log: GenerationLog) {
  const fileName = `${log.date}.json`;
  const filePath = path.join(LOGS_DIR, fileName);
  
  // Ensure directory exists
  if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true });
  }
  
  // Append to existing log if present
  let existingLogs: GenerationLog[] = [];
  if (fs.existsSync(filePath)) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      existingLogs = JSON.parse(content);
      if (!Array.isArray(existingLogs)) existingLogs = [existingLogs];
    } catch {
      existingLogs = [];
    }
  }
  
  existingLogs.push(log);
  fs.writeFileSync(filePath, JSON.stringify(existingLogs, null, 2));
  console.log(`\n📝 Log written to: ${filePath}`);
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  const { days, dryRun, retryFailed, archetype, ageBucket } = parseArgs();
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║           🚀 CURIOUS KELLY DAILY GENERATION ENGINE           ║
╚══════════════════════════════════════════════════════════════╝

Target Days:   ${days.join(', ')}
Archetype:     ${archetype}
Age Bucket:    ${ageBucket}
Dry Run:       ${dryRun ? 'YES (no actual generation)' : 'NO'}
Retry Failed:  ${retryFailed ? 'YES' : 'NO'}
`);

  if (!HEYGEN_API_KEY && !dryRun) {
    console.error('❌ HEYGEN_API_KEY not set. Use --dry-run to preview.');
    process.exit(1);
  }
  if (!ELEVENLABS_API_KEY && !dryRun) {
    console.error('❌ ELEVENLABS_API_KEY not set. Use --dry-run to preview.');
    process.exit(1);
  }

  const startedAt = new Date().toISOString();
  const date = startedAt.split('T')[0];
  
  // Get current costs
  const costs = await getTodayCosts();
  console.log(`💰 Today's usage so far:`);
  console.log(`   HeyGen credits: ${costs.heygen_credits}/${DAILY_LIMITS.heygen_credits}`);
  console.log(`   ElevenLabs chars: ${costs.elevenlabs_characters}/${DAILY_LIMITS.elevenlabs_characters}`);
  console.log(`   Estimated USD: $${costs.estimated_usd.toFixed(2)}/$${DAILY_LIMITS.max_usd}`);
  
  // Check limits before starting
  const limitCheck = checkLimits(costs);
  if (!limitCheck.ok) {
    console.error(`\n🛑 ${limitCheck.reason}`);
    await alert('DAILY_LIMIT_REACHED', { 
      spent: costs.estimated_usd, 
      limit: DAILY_LIMITS.max_usd 
    });
    process.exit(1);
  }
  
  await alert('GENERATION_STARTED', { days });
  
  const log: GenerationLog = {
    date,
    started_at: startedAt,
    completed_at: '',
    target_days: days,
    archetype,
    age_bucket: ageBucket,
    dry_run: dryRun,
    results: {},
    total_cost_usd: 0,
    total_heygen_credits: 0,
    total_elevenlabs_characters: 0,
    errors: [],
    aborted: false,
  };
  
  let aborted = false;
  
  for (const day of days) {
    // Check limits before each day
    const currentCosts = await getTodayCosts();
    const check = checkLimits(currentCosts);
    if (!check.ok) {
      console.log(`\n🛑 ${check.reason}`);
      log.aborted = true;
      log.abort_reason = check.reason;
      aborted = true;
      await alert('DAILY_LIMIT_REACHED', { 
        spent: currentCosts.estimated_usd, 
        limit: DAILY_LIMITS.max_usd 
      });
      break;
    }
    
    const result = await generateDay(day, archetype, ageBucket, currentCosts, dryRun, retryFailed);
    log.results[day] = result;
    log.total_heygen_credits += result.heygen_cost_credits;
    log.total_elevenlabs_characters += result.elevenlabs_characters;
    log.errors.push(...result.errors);
  }
  
  // Calculate total cost
  const finalCosts = await getTodayCosts();
  log.total_cost_usd = finalCosts.estimated_usd - costs.estimated_usd;
  log.completed_at = new Date().toISOString();
  
  // Write log
  writeLog(log);
  
  // Summary
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                        📊 SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝
`);
  
  for (const [day, result] of Object.entries(log.results)) {
    const icon = result.status === 'success' ? '✅' :
                 result.status === 'partial' ? '🟡' :
                 result.status === 'failed' ? '❌' : '⏭️';
    console.log(`Day ${day}: ${icon} ${result.status} (${result.videos_generated} generated, ${result.videos_skipped} skipped, ${result.videos_failed} failed)`);
  }
  
  console.log(`
Total Cost:        $${log.total_cost_usd.toFixed(2)}
HeyGen Credits:    ${log.total_heygen_credits}
ElevenLabs Chars:  ${log.total_elevenlabs_characters}
Duration:          ${Math.round((new Date(log.completed_at).getTime() - new Date(log.started_at).getTime()) / 1000)}s
`);

  if (!aborted && !dryRun) {
    const successDays = Object.entries(log.results)
      .filter(([, r]) => r.status === 'success' || r.status === 'skipped')
      .map(([d]) => parseInt(d));
    
    if (successDays.length > 0) {
      await alert('GENERATION_COMPLETE', { 
        days: successDays, 
        cost: log.total_cost_usd 
      });
      
      console.log(`
✨ Next steps:
   1. Review days at: public/admin/generation-status.html
   2. Approve each day: npx tsx scripts/approve-day.ts --day=<N>
`);
    }
  }
}

main().catch(async (error) => {
  console.error('❌ Fatal error:', error);
  await alert('SYSTEM_ERROR', { error: error.message });
  process.exit(1);
});



