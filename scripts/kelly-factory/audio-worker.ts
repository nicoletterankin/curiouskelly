#!/usr/bin/env npx tsx
/**
 * Kelly Factory - Audio Generation Worker
 * 
 * Generates audio from scripts using ElevenLabs API
 * Processes script_ready assets from kelly_lesson_assets
 * 
 * Usage:
 *   npx tsx scripts/kelly-factory/audio-worker.ts              # Process all script_ready
 *   npx tsx scripts/kelly-factory/audio-worker.ts --day=1      # Process specific day
 *   npx tsx scripts/kelly-factory/audio-worker.ts --limit=10   # Limit batch size
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
  
  // Kelly voice IDs by age group
  voiceIds: {
    5: 'XB0fDUnXU5powFXDhCwa',   // Young playful
    8: 'XB0fDUnXU5powFXDhCwa',   // Kid-friendly
    16: 'XB0fDUnXU5powFXDhCwa',  // Teen
    25: 'XB0fDUnXU5powFXDhCwa',  // Young adult
    35: 'XB0fDUnXU5powFXDhCwa',  // Adult (default Kelly)
    50: 'XB0fDUnXU5powFXDhCwa',  // Mature
    70: 'XB0fDUnXU5powFXDhCwa',  // Elder
    102: 'XB0fDUnXU5powFXDhCwa', // Wisdom
  } as Record<number, string>,
  
  defaultVoiceId: 'XB0fDUnXU5powFXDhCwa', // Kelly's main voice
  
  // Voice settings by age
  voiceSettings: {
    5: { stability: 0.5, similarity_boost: 0.8, style: 0.6 },
    8: { stability: 0.55, similarity_boost: 0.8, style: 0.5 },
    16: { stability: 0.6, similarity_boost: 0.85, style: 0.4 },
    35: { stability: 0.65, similarity_boost: 0.85, style: 0.3 },
    50: { stability: 0.7, similarity_boost: 0.9, style: 0.25 },
    70: { stability: 0.75, similarity_boost: 0.9, style: 0.2 },
  } as Record<number, { stability: number; similarity_boost: number; style: number }>,
  
  defaultBatchSize: 20,
  model: 'eleven_turbo_v2',
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

// ═══════════════════════════════════════════════════════════════════════════
// ELEVENLABS GENERATION
// ═══════════════════════════════════════════════════════════════════════════

async function generateAudio(asset: LessonAsset): Promise<{
  success: boolean;
  audioUrl?: string;
  duration?: number;
  error?: string;
}> {
  const voiceId = CONFIG.voiceIds[asset.age_group] || CONFIG.defaultVoiceId;
  const settings = CONFIG.voiceSettings[asset.age_group] || CONFIG.voiceSettings[35];

  console.log(`   🎤 Generating audio with ElevenLabs...`);
  
  try {
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': CONFIG.elevenLabsKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: asset.script,
          model_id: CONFIG.model,
          voice_settings: settings,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.text();
      return { success: false, error: `ElevenLabs error: ${error}` };
    }

    // Get audio buffer
    const audioBuffer = Buffer.from(await response.arrayBuffer());
    
    // Upload to Supabase Storage
    const storagePath = `audio/day-${asset.day_number}/${asset.phase}-age${asset.age_group}-${asset.language}.mp3`;
    console.log(`   📤 Uploading to ${storagePath}...`);
    
    const { error: uploadError } = await supabase.storage
      .from('kelly-audio')
      .upload(storagePath, audioBuffer, { 
        contentType: 'audio/mpeg', 
        upsert: true 
      });

    if (uploadError) {
      // Try kelly-templates bucket as fallback
      const { error: fallbackError } = await supabase.storage
        .from('kelly-templates')
        .upload(`audio/${storagePath}`, audioBuffer, { 
          contentType: 'audio/mpeg', 
          upsert: true 
        });
      
      if (fallbackError) {
        return { success: false, error: `Upload error: ${uploadError.message}` };
      }
      
      const { data: urlData } = supabase.storage
        .from('kelly-templates')
        .getPublicUrl(`audio/${storagePath}`);
      
      return { success: true, audioUrl: urlData.publicUrl };
    }

    const { data: urlData } = supabase.storage
      .from('kelly-audio')
      .getPublicUrl(storagePath);

    // Estimate duration (rough: ~150 words/minute = 2.5 words/second)
    const wordCount = asset.script.split(/\s+/).length;
    const estimatedDuration = wordCount / 2.5;

    return { 
      success: true, 
      audioUrl: urlData.publicUrl,
      duration: estimatedDuration 
    };
    
  } catch (error) {
    return { success: false, error: `Generation error: ${(error as Error).message}` };
  }
}

async function processAsset(asset: LessonAsset): Promise<boolean> {
  console.log(`\n📦 Day ${asset.day_number} | ${asset.phase} | Age ${asset.age_group} | ${asset.language}`);
  console.log(`   Script: "${asset.script.substring(0, 60)}..."`);

  const result = await generateAudio(asset);

  if (result.success && result.audioUrl) {
    // Update registry
    const { error: updateError } = await supabase
      .from('kelly_lesson_assets')
      .update({
        audio_url: result.audioUrl,
        audio_duration: result.duration,
        audio_source: 'elevenlabs',
        status: 'audio_ready',
        updated_at: new Date().toISOString(),
      })
      .eq('id', asset.id);

    if (updateError) {
      console.log(`   ⚠️  Registry update error: ${updateError.message}`);
    } else {
      console.log(`   ✅ Audio ready: ${result.audioUrl.substring(0, 60)}...`);
    }
    return true;
  }

  console.log(`   ❌ Failed: ${result.error}`);
  await supabase
    .from('kelly_lesson_assets')
    .update({ 
      status: 'error', 
      error_message: result.error,
      updated_at: new Date().toISOString() 
    })
    .eq('id', asset.id);
    
  return false;
}

async function getAssetsToProcess(options: {
  day?: number;
  limit?: number;
}): Promise<LessonAsset[]> {
  let query = supabase
    .from('kelly_lesson_assets')
    .select('*')
    .eq('status', 'script_ready')
    .not('script', 'is', null);

  if (options.day) {
    query = query.eq('day_number', options.day);
  }

  query = query
    .order('day_number', { ascending: true })
    .limit(options.limit || CONFIG.defaultBatchSize);

  const { data, error } = await query;

  if (error) {
    console.error('Error fetching assets:', error);
    return [];
  }

  return data || [];
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║           🎤 KELLY FACTORY - AUDIO GENERATION WORKER                     ║
╚══════════════════════════════════════════════════════════════════════════╝
`);

  // Parse CLI args
  const args = process.argv.slice(2);
  const options: { day?: number; limit?: number } = {};

  for (const arg of args) {
    if (arg.startsWith('--day=')) options.day = parseInt(arg.split('=')[1]);
    if (arg.startsWith('--limit=')) options.limit = parseInt(arg.split('=')[1]);
  }

  // Check ElevenLabs
  console.log(`ElevenLabs API: ${CONFIG.elevenLabsKey ? '✅ Configured' : '❌ Not configured'}\n`);

  if (!CONFIG.elevenLabsKey) {
    console.error('❌ ELEVENLABS_API_KEY not set.');
    process.exit(1);
  }

  // Get assets
  const assets = await getAssetsToProcess(options);
  
  if (assets.length === 0) {
    console.log('✨ No script_ready assets to process.');
    return;
  }

  console.log(`Found ${assets.length} assets with scripts\n`);

  // Process
  let success = 0;
  let failed = 0;

  for (const asset of assets) {
    const result = await processAsset(asset);
    if (result) success++;
    else failed++;
  }

  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║                              📊 SUMMARY                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Processed: ${assets.length} assets
  ✅ Success: ${success}
  ❌ Failed: ${failed}

${success > 0 ? '🎉 Audio files generated and registered!' : ''}
`);
}

main().catch(console.error);
