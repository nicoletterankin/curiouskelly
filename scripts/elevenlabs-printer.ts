#!/usr/bin/env npx tsx
/**
 * ElevenLabs Audio Printer
 * 
 * Generates audio for ALL pending kelly_lesson_assets using ElevenLabs
 * Can also populate scripts from lesson_atoms if needed
 * 
 * Usage:
 *   npx tsx scripts/elevenlabs-printer.ts                    # Process all script_ready
 *   npx tsx scripts/elevenlabs-printer.ts --populate         # Populate scripts from lesson_atoms first
 *   npx tsx scripts/elevenlabs-printer.ts --limit=100        # Limit batch size
 *   npx tsx scripts/elevenlabs-printer.ts --day=1-10         # Process day range
 *   npx tsx scripts/elevenlabs-printer.ts --status           # Show status only
 *   npx tsx scripts/elevenlabs-printer.ts --dry-run          # Preview without generating
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  supabaseUrl: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '',
  supabaseKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '',
  elevenLabsKey: process.env.ELEVENLABS_API_KEY || process.env.ELEVEN_LABS_API_KEY || '',
  
  // Kelly's trained voice (primary)
  kellyVoiceId: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  
  // Voice settings tuned for Kelly
  voiceSettings: {
    stability: 0.65,
    similarity_boost: 0.85,
    style: 0.3,
    use_speaker_boost: true,
  },
  
  // Model - turbo for speed, multilingual for quality
  model: 'eleven_turbo_v2',
  outputFormat: 'mp3_44100_128',
  
  // Rate limiting
  requestDelayMs: 300,  // ms between requests to avoid rate limits
  batchSize: 50,        // Assets per batch
  
  // Mapping archetypes to age groups for script population
  archetypeToAge: {
    'The Explorer': 35,
    'The Scientist': 35,
    'The Survivor': 35,
    'The MacGyver': 35,
    'The Rebel': 16,
    'The Provider': 50,
    'The Storyteller': 8,
    'The Strategist': 35,
    'The Mystic': 70,
    'The Architect': 35,
    'The Diplomat': 50,
    'The Empath': 35,
  } as Record<string, number>,
  
  // Phase mapping (lesson_atoms uses Hook/Fact1/etc, kelly_lesson_assets uses hook/story/etc)
  phaseMapping: {
    'Hook': 'hook',
    'Fact1': 'story',
    'Fact2': 'wonder',
    'Fact3': 'action',
    'Wisdom': 'wisdom',
  } as Record<string, string>,
};

const supabase = createClient(CONFIG.supabaseUrl, CONFIG.supabaseKey);

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

interface LessonAsset {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  language: string;
  script: string;
  status: string;
}

interface GenerationStats {
  processed: number;
  success: number;
  failed: number;
  charactersUsed: number;
  errors: string[];
  startTime: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// ELEVENLABS API
// ═══════════════════════════════════════════════════════════════════════════

async function generateAudioWithElevenLabs(
  script: string,
  voiceId: string = CONFIG.kellyVoiceId
): Promise<{ buffer: Buffer; characters: number } | { error: string }> {
  try {
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=${CONFIG.outputFormat}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': CONFIG.elevenLabsKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: script,
          model_id: CONFIG.model,
          voice_settings: CONFIG.voiceSettings,
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { error: `ElevenLabs ${response.status}: ${errorText}` };
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    return { buffer: audioBuffer, characters: script.length };
    
  } catch (error) {
    return { error: `Network error: ${(error as Error).message}` };
  }
}

async function getElevenLabsUsage(): Promise<{ remaining: number; total: number } | null> {
  try {
    const response = await fetch('https://api.elevenlabs.io/v1/user/subscription', {
      headers: { 'xi-api-key': CONFIG.elevenLabsKey },
    });
    
    if (response.ok) {
      const data = await response.json();
      return {
        remaining: data.character_limit - data.character_count,
        total: data.character_limit,
      };
    }
    return null;
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SUPABASE STORAGE
// ═══════════════════════════════════════════════════════════════════════════

async function uploadToStorage(
  buffer: Buffer,
  asset: LessonAsset
): Promise<string | null> {
  const storagePath = `audio/2026/${asset.language}/day-${String(asset.day_number).padStart(3, '0')}/${asset.phase}-age${asset.age_group}.mp3`;
  
  // Use kelly-templates bucket (primary storage)
  const { error: uploadError } = await supabase.storage
    .from('kelly-templates')
    .upload(storagePath, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });

  if (!uploadError) {
    const { data: urlData } = supabase.storage
      .from('kelly-templates')
      .getPublicUrl(storagePath);
    return urlData.publicUrl;
  }

  // Try kelly-videos as alternative
  const { error: fallbackError } = await supabase.storage
    .from('kelly-videos')
    .upload(`audio/${storagePath}`, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });

  if (!fallbackError) {
    const { data: urlData } = supabase.storage
      .from('kelly-videos')
      .getPublicUrl(`audio/${storagePath}`);
    return urlData.publicUrl;
  }

  console.error(`   ❌ Storage upload failed: ${uploadError?.message}`);
  return null;
}

// ═══════════════════════════════════════════════════════════════════════════
// SCRIPT POPULATION (from lesson_atoms)
// ═══════════════════════════════════════════════════════════════════════════

async function populateScriptsFromLessonAtoms(
  dayRange?: { start: number; end: number },
  limit?: number
): Promise<number> {
  console.log('\n📝 Populating scripts from lesson_atoms...\n');
  
  // Use SQL to efficiently batch-populate scripts
  // Maps age_group → archetype: 5,8 → Storyteller, 16 → Rebel, 35 → Explorer, 50,70 → Provider/Mystic
  const ageToArchetype: Record<number, string> = {
    5: 'The Storyteller',
    8: 'The Storyteller', 
    16: 'The Rebel',
    25: 'The Explorer',
    35: 'The Explorer',
    50: 'The Provider',
    70: 'The Mystic',
    102: 'The Mystic',
  };
  
  // Phase mapping: kelly_lesson_assets phase → lesson_atoms phase
  const phaseToAtom: Record<string, string> = {
    'hook': 'Hook',
    'story': 'Fact1',
    'wonder': 'Fact2',
    'action': 'Fact3',
    'wisdom': 'Wisdom',
  };
  
  // Get pending assets that need scripts
  let query = supabase
    .from('kelly_lesson_assets')
    .select('id, day_number, phase, age_group, language')
    .eq('status', 'pending')
    .is('script', null);
  
  if (dayRange) {
    query = query.gte('day_number', dayRange.start).lte('day_number', dayRange.end);
  }
  
  query = query.order('day_number').limit(limit || 1000);
  
  const { data: pendingAssets, error } = await query;
  
  if (error || !pendingAssets?.length) {
    console.log('   No pending assets found or error:', error?.message);
    return 0;
  }
  
  console.log(`   Found ${pendingAssets.length} pending assets to populate`);
  
  // Get all core_lessons for day_number mapping
  const { data: coreLessons } = await supabase
    .from('core_lessons')
    .select('id, day_number');
  
  const dayToLessonId = new Map(coreLessons?.map(cl => [cl.day_number, cl.id]) || []);
  
  let populated = 0;
  let failed = 0;
  
  for (const asset of pendingAssets) {
    const archetype = ageToArchetype[asset.age_group] || 'The Explorer';
    const atomPhase = phaseToAtom[asset.phase] || 'Hook';
    const coreLessonId = dayToLessonId.get(asset.day_number);
    
    if (!coreLessonId) {
      failed++;
      continue;
    }
    
    // Fetch script from lesson_atoms
    const { data: atomData } = await supabase
      .from('lesson_atoms')
      .select('content')
      .eq('archetype', archetype)
      .eq('phase', atomPhase)
      .eq('core_lesson_id', coreLessonId)
      .single();
    
    const script = atomData?.content?.script;
    
    if (script) {
      const { error: updateError } = await supabase
        .from('kelly_lesson_assets')
        .update({
          script: script,
          status: 'script_ready',
          updated_at: new Date().toISOString(),
        })
        .eq('id', asset.id);
      
      if (!updateError) {
        populated++;
        if (populated % 100 === 0) {
          console.log(`   📝 Populated ${populated}/${pendingAssets.length} scripts...`);
        }
      } else {
        failed++;
      }
    } else {
      failed++;
    }
  }
  
  console.log(`   ✅ Populated ${populated} scripts (${failed} failed/skipped)\n`);
  return populated;
}

// ═══════════════════════════════════════════════════════════════════════════
// AUDIO GENERATION PIPELINE
// ═══════════════════════════════════════════════════════════════════════════

async function processAsset(
  asset: LessonAsset,
  stats: GenerationStats,
  dryRun: boolean
): Promise<boolean> {
  const prefix = `[Day ${asset.day_number}/${asset.phase}/Age ${asset.age_group}]`;
  
  if (dryRun) {
    console.log(`${prefix} Would generate: ${asset.script.length} chars`);
    stats.charactersUsed += asset.script.length;
    stats.success++;
    return true;
  }
  
  console.log(`${prefix} Generating audio (${asset.script.length} chars)...`);
  
  // Generate audio
  const result = await generateAudioWithElevenLabs(asset.script);
  
  if ('error' in result) {
    console.log(`${prefix} ❌ ${result.error}`);
    stats.errors.push(`${prefix}: ${result.error}`);
    stats.failed++;
    
    await supabase
      .from('kelly_lesson_assets')
      .update({
        status: 'error',
        error_message: result.error,
        updated_at: new Date().toISOString(),
      })
      .eq('id', asset.id);
    
    return false;
  }
  
  // Upload to storage
  const audioUrl = await uploadToStorage(result.buffer, asset);
  
  if (!audioUrl) {
    stats.errors.push(`${prefix}: Storage upload failed`);
    stats.failed++;
    return false;
  }
  
  // Update database
  const wordCount = asset.script.split(/\s+/).length;
  const estimatedDuration = wordCount / 2.5; // ~150 words/minute
  
  await supabase
    .from('kelly_lesson_assets')
    .update({
      audio_url: audioUrl,
      audio_duration: estimatedDuration,
      audio_source: 'elevenlabs',
      status: 'audio_ready',
      updated_at: new Date().toISOString(),
    })
    .eq('id', asset.id);
  
  console.log(`${prefix} ✅ Uploaded: ${audioUrl.split('/').slice(-2).join('/')}`);
  
  stats.charactersUsed += result.characters;
  stats.success++;
  
  return true;
}

async function generateAudioBatch(
  dayRange?: { start: number; end: number },
  limit?: number,
  dryRun: boolean = false
): Promise<GenerationStats> {
  const stats: GenerationStats = {
    processed: 0,
    success: 0,
    failed: 0,
    charactersUsed: 0,
    errors: [],
    startTime: Date.now(),
  };
  
  // Get assets with scripts ready
  let query = supabase
    .from('kelly_lesson_assets')
    .select('*')
    .in('status', ['script_ready', 'pending'])
    .not('script', 'is', null);
  
  if (dayRange) {
    query = query.gte('day_number', dayRange.start).lte('day_number', dayRange.end);
  }
  
  query = query
    .order('day_number', { ascending: true })
    .order('phase', { ascending: true })
    .limit(limit || CONFIG.batchSize);
  
  const { data: assets, error } = await query;
  
  if (error) {
    console.error('Error fetching assets:', error);
    return stats;
  }
  
  if (!assets?.length) {
    console.log('✨ No assets with scripts to process.');
    return stats;
  }
  
  console.log(`\n🎤 Processing ${assets.length} assets${dryRun ? ' (DRY RUN)' : ''}...\n`);
  
  for (const asset of assets) {
    stats.processed++;
    await processAsset(asset as LessonAsset, stats, dryRun);
    
    // Rate limiting
    if (!dryRun && stats.processed < assets.length) {
      await new Promise(r => setTimeout(r, CONFIG.requestDelayMs));
    }
    
    // Progress update every 10
    if (stats.processed % 10 === 0) {
      const elapsed = ((Date.now() - stats.startTime) / 1000).toFixed(1);
      console.log(`\n📊 Progress: ${stats.processed}/${assets.length} | ${stats.success}✅ ${stats.failed}❌ | ${stats.charactersUsed.toLocaleString()} chars | ${elapsed}s\n`);
    }
  }
  
  return stats;
}

// ═══════════════════════════════════════════════════════════════════════════
// STATUS DISPLAY
// ═══════════════════════════════════════════════════════════════════════════

async function showStatus() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║           🖨️  ELEVENLABS AUDIO PRINTER - STATUS                          ║
╚══════════════════════════════════════════════════════════════════════════╝
`);

  // Database status
  const { data: statusCounts } = await supabase
    .from('kelly_lesson_assets')
    .select('status')
    .then(result => {
      const counts: Record<string, number> = {};
      result.data?.forEach(r => {
        counts[r.status] = (counts[r.status] || 0) + 1;
      });
      return { data: counts };
    });
  
  // Script availability
  const { data: scriptCounts } = await supabase
    .from('kelly_lesson_assets')
    .select('status, script')
    .not('script', 'is', null)
    .then(result => {
      return { data: { count: result.data?.length || 0 } };
    });
  
  const { data: charCount } = await supabase
    .rpc('get_total_script_chars');
  
  console.log('Database Status:');
  console.log('┌─────────────────┬──────────┐');
  console.log('│ Status          │ Count    │');
  console.log('├─────────────────┼──────────┤');
  
  const statuses = ['pending', 'script_ready', 'audio_ready', 'complete', 'error'];
  for (const status of statuses) {
    const icon = status === 'complete' ? '✅' : status === 'audio_ready' ? '🎵' : status === 'script_ready' ? '📝' : status === 'error' ? '❌' : '⏳';
    console.log(`│ ${icon} ${status.padEnd(13)} │ ${String(statusCounts?.[status] || 0).padStart(8)} │`);
  }
  console.log('└─────────────────┴──────────┘');
  
  console.log(`\nAssets with scripts: ${scriptCounts?.count || 0}`);
  
  // ElevenLabs quota
  const usage = await getElevenLabsUsage();
  if (usage) {
    const pct = ((usage.remaining / usage.total) * 100).toFixed(1);
    console.log(`
ElevenLabs Quota:
  Total:     ${usage.total.toLocaleString()} characters
  Remaining: ${usage.remaining.toLocaleString()} characters (${pct}%)
`);
  } else {
    console.log('\n⚠️  Could not fetch ElevenLabs quota');
  }
  
  // Estimate
  if (scriptCounts?.count && usage) {
    const avgCharsPerScript = 200; // Conservative estimate
    const neededChars = (scriptCounts.count as number) * avgCharsPerScript;
    const canProcess = Math.floor(usage.remaining / avgCharsPerScript);
    console.log(`Estimate: Can process ~${canProcess.toLocaleString()} assets with current quota`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

function parseArgs(): {
  populate: boolean;
  limit?: number;
  dayRange?: { start: number; end: number };
  status: boolean;
  dryRun: boolean;
} {
  const args = process.argv.slice(2);
  
  const options = {
    populate: args.includes('--populate'),
    status: args.includes('--status'),
    dryRun: args.includes('--dry-run'),
    limit: undefined as number | undefined,
    dayRange: undefined as { start: number; end: number } | undefined,
  };
  
  for (const arg of args) {
    if (arg.startsWith('--limit=')) {
      options.limit = parseInt(arg.split('=')[1]);
    }
    if (arg.startsWith('--day=')) {
      const range = arg.split('=')[1];
      if (range.includes('-')) {
        const [start, end] = range.split('-').map(Number);
        options.dayRange = { start, end };
      } else {
        const day = parseInt(range);
        options.dayRange = { start: day, end: day };
      }
    }
  }
  
  return options;
}

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║           🖨️  ELEVENLABS AUDIO PRINTER                                   ║
║           Curious Kelly Voice Generation Pipeline                        ║
╚══════════════════════════════════════════════════════════════════════════╝
`);

  // Validate config
  if (!CONFIG.elevenLabsKey) {
    console.error('❌ ELEVENLABS_API_KEY not set. Add it to your .env file.');
    process.exit(1);
  }
  
  if (!CONFIG.supabaseUrl || !CONFIG.supabaseKey) {
    console.error('❌ Supabase credentials not set. Check your .env file.');
    process.exit(1);
  }

  const options = parseArgs();
  
  // Status only mode
  if (options.status) {
    await showStatus();
    return;
  }
  
  // Show current quota
  const usage = await getElevenLabsUsage();
  if (usage) {
    console.log(`ElevenLabs Quota: ${usage.remaining.toLocaleString()} / ${usage.total.toLocaleString()} characters remaining\n`);
  }
  
  // Populate scripts if requested
  if (options.populate) {
    await populateScriptsFromLessonAtoms(options.dayRange, options.limit);
  }
  
  // Generate audio
  const stats = await generateAudioBatch(options.dayRange, options.limit, options.dryRun);
  
  // Final report
  const elapsed = ((Date.now() - stats.startTime) / 1000).toFixed(1);
  
  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║                              📊 FINAL REPORT                             ║
╚══════════════════════════════════════════════════════════════════════════╝

Assets processed: ${stats.processed}
  ✅ Success: ${stats.success}
  ❌ Failed: ${stats.failed}

Characters used: ${stats.charactersUsed.toLocaleString()}
Time elapsed: ${elapsed}s
${options.dryRun ? '\n⚠️  DRY RUN - No audio was actually generated' : ''}
`);

  if (stats.errors.length > 0) {
    console.log('Errors:');
    stats.errors.slice(0, 10).forEach(e => console.log(`  - ${e}`));
    if (stats.errors.length > 10) {
      console.log(`  ... and ${stats.errors.length - 10} more`);
    }
  }
  
  // Show updated quota
  const finalUsage = await getElevenLabsUsage();
  if (finalUsage) {
    console.log(`\nRemaining ElevenLabs quota: ${finalUsage.remaining.toLocaleString()} characters`);
  }
}

main().catch(console.error);
