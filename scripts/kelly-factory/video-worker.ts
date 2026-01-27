#!/usr/bin/env npx tsx
/**
 * Kelly Factory - Video Generation Worker
 * 
 * Processes audio_ready assets from kelly_lesson_assets and generates lip-synced videos
 * using multiple backends: Fal.ai SadTalker (fast), HeyGen (quality), or Local SadTalker (free)
 * 
 * Usage:
 *   npx tsx scripts/kelly-factory/video-worker.ts              # Process all audio_ready
 *   npx tsx scripts/kelly-factory/video-worker.ts --day=39     # Process specific day
 *   npx tsx scripts/kelly-factory/video-worker.ts --limit=5    # Limit batch size
 *   npx tsx scripts/kelly-factory/video-worker.ts --backend=fal # Force backend
 */

import 'dotenv/config';
import { fal } from '@fal-ai/client';
import { createClient } from '@supabase/supabase-js';

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  // Supabase
  supabaseUrl: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '',
  supabaseKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '',
  
  // Kelly reference images by age group
  kellyImages: {
    default: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/kelly_1765361262640.png',
    // Add age-specific images when available
  } as Record<string | number, string>,
  
  // Backend priority (first available wins)
  backendPriority: ['fal', 'heygen', 'local'] as const,
  
  // API Keys
  falKey: process.env.FAL_KEY,
  heygenKey: process.env.HEYGEN_API_KEY,
  
  // Limits
  maxConcurrent: 3,
  defaultBatchSize: 10,
};

// Initialize clients
const supabase = createClient(CONFIG.supabaseUrl, CONFIG.supabaseKey);
if (CONFIG.falKey) fal.config({ credentials: CONFIG.falKey });

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

interface LessonAsset {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  language: string;
  audio_url: string;
  audio_duration: number | null;
  status: string;
}

type Backend = 'fal' | 'heygen' | 'local';

interface GenerationResult {
  success: boolean;
  videoUrl?: string;
  backend?: Backend;
  duration?: number;
  error?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKEND: FAL.AI SADTALKER (Cloud - Fast)
// ═══════════════════════════════════════════════════════════════════════════

async function generateWithFal(asset: LessonAsset): Promise<GenerationResult> {
  if (!CONFIG.falKey) {
    return { success: false, error: 'FAL_KEY not configured' };
  }

  const kellyImage = CONFIG.kellyImages[asset.age_group] || CONFIG.kellyImages.default;
  
  console.log(`   🎬 Fal.ai SadTalker processing...`);
  const startTime = Date.now();
  
  try {
    const result = await fal.subscribe('fal-ai/sadtalker', {
      input: {
        source_image_url: kellyImage,
        driven_audio_url: asset.audio_url,
        still: true,  // Minimal head motion for teaching
        enhancer: 'gfpgan',
        preprocess: 'full',
      },
      logs: false,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') {
          process.stdout.write('.');
        }
      }
    });

    const videoUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
    
    if (!videoUrl) {
      return { success: false, error: 'No video URL in Fal response' };
    }

    const duration = (Date.now() - startTime) / 1000;
    console.log(`\n   ✅ Fal completed in ${duration.toFixed(1)}s`);
    
    return { success: true, videoUrl, backend: 'fal', duration };
    
  } catch (error) {
    return { success: false, error: `Fal error: ${(error as Error).message}` };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKEND: HEYGEN (Cloud - High Quality)
// ═══════════════════════════════════════════════════════════════════════════

async function generateWithHeyGen(asset: LessonAsset): Promise<GenerationResult> {
  if (!CONFIG.heygenKey) {
    return { success: false, error: 'HEYGEN_API_KEY not configured' };
  }

  console.log(`   🎬 HeyGen processing...`);
  const startTime = Date.now();
  
  try {
    // Submit video generation request
    const submitResp = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Api-Key': CONFIG.heygenKey,
      },
      body: JSON.stringify({
        video_inputs: [{
          character: {
            type: 'talking_photo',
            talking_photo_id: 'kelly_default', // Configure with actual ID
          },
          voice: {
            type: 'audio',
            audio_url: asset.audio_url,
          },
        }],
        dimension: { width: 512, height: 512 },
      }),
    });

    const submitData = await submitResp.json();
    
    if (submitData.error) {
      return { success: false, error: `HeyGen submit: ${submitData.error.message}` };
    }

    const videoId = submitData.data?.video_id;
    if (!videoId) {
      return { success: false, error: 'No video_id in HeyGen response' };
    }

    // Poll for completion
    console.log(`   ⏳ Waiting for HeyGen (${videoId})...`);
    const maxWait = 300000; // 5 minutes
    const pollInterval = 10000;
    const deadline = Date.now() + maxWait;

    while (Date.now() < deadline) {
      const statusResp = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
        { headers: { 'X-Api-Key': CONFIG.heygenKey } }
      );
      const statusData = await statusResp.json();
      const status = statusData.data?.status;

      if (status === 'completed') {
        const videoUrl = statusData.data?.video_url;
        const duration = (Date.now() - startTime) / 1000;
        console.log(`   ✅ HeyGen completed in ${duration.toFixed(1)}s`);
        return { success: true, videoUrl, backend: 'heygen', duration };
      }

      if (status === 'failed') {
        return { success: false, error: `HeyGen failed: ${statusData.data?.error}` };
      }

      process.stdout.write('.');
      await new Promise(r => setTimeout(r, pollInterval));
    }

    return { success: false, error: 'HeyGen timeout' };
    
  } catch (error) {
    return { success: false, error: `HeyGen error: ${(error as Error).message}` };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// UPLOAD & REGISTRY UPDATE
// ═══════════════════════════════════════════════════════════════════════════

async function uploadAndRegister(asset: LessonAsset, result: GenerationResult): Promise<string | null> {
  if (!result.success || !result.videoUrl) return null;

  try {
    // Download video
    console.log('   📥 Downloading video...');
    const response = await fetch(result.videoUrl);
    const buffer = Buffer.from(await response.arrayBuffer());

    // Upload to Supabase Storage
    const storagePath = `lipsync/day-${asset.day_number}/${asset.phase}-age${asset.age_group}-${asset.language}.mp4`;
    console.log(`   📤 Uploading to ${storagePath}...`);
    
    const { error: uploadError } = await supabase.storage
      .from('kelly-videos')
      .upload(storagePath, buffer, { contentType: 'video/mp4', upsert: true });

    if (uploadError) {
      console.log(`   ⚠️  Upload error: ${uploadError.message}`);
      // Still use the original URL
      return result.videoUrl;
    }

    const { data: urlData } = supabase.storage
      .from('kelly-videos')
      .getPublicUrl(storagePath);

    const publicUrl = urlData.publicUrl;

    // Update registry
    const { error: updateError } = await supabase
      .from('kelly_lesson_assets')
      .update({
        video_url: publicUrl,
        video_source: result.backend,
        video_duration: asset.audio_duration, // Video duration ≈ audio duration
        status: 'complete',
        updated_at: new Date().toISOString(),
      })
      .eq('id', asset.id);

    if (updateError) {
      console.log(`   ⚠️  Registry update error: ${updateError.message}`);
    }

    return publicUrl;
    
  } catch (error) {
    console.log(`   ❌ Upload error: ${(error as Error).message}`);
    return result.videoUrl; // Return original URL as fallback
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WORKER LOGIC
// ═══════════════════════════════════════════════════════════════════════════

async function processAsset(asset: LessonAsset, forceBackend?: Backend): Promise<boolean> {
  console.log(`\n📦 Day ${asset.day_number} | ${asset.phase} | Age ${asset.age_group} | ${asset.language}`);
  console.log(`   Audio: ${asset.audio_url?.substring(0, 60)}...`);

  // Determine backend order
  const backends = forceBackend 
    ? [forceBackend] 
    : CONFIG.backendPriority;

  let result: GenerationResult = { success: false };

  for (const backend of backends) {
    console.log(`   🔄 Trying ${backend}...`);
    
    switch (backend) {
      case 'fal':
        result = await generateWithFal(asset);
        break;
      case 'heygen':
        result = await generateWithHeyGen(asset);
        break;
      case 'local':
        // Local SadTalker - skip in cloud worker (use batch script)
        result = { success: false, error: 'Local not available in worker' };
        break;
    }

    if (result.success) {
      const publicUrl = await uploadAndRegister(asset, result);
      if (publicUrl) {
        console.log(`   ✅ Complete: ${publicUrl.substring(0, 60)}...`);
        return true;
      }
    } else {
      console.log(`   ⚠️  ${backend}: ${result.error}`);
    }
  }

  // Mark as error
  await supabase
    .from('kelly_lesson_assets')
    .update({ 
      status: 'error', 
      error_message: result.error,
      updated_at: new Date().toISOString() 
    })
    .eq('id', asset.id);

  console.log(`   ❌ All backends failed`);
  return false;
}

async function getAssetsToProcess(options: {
  day?: number;
  limit?: number;
  ageGroup?: number;
}): Promise<LessonAsset[]> {
  let query = supabase
    .from('kelly_lesson_assets')
    .select('*')
    .eq('status', 'audio_ready')
    .not('audio_url', 'is', null);

  if (options.day) {
    query = query.eq('day_number', options.day);
  }
  if (options.ageGroup) {
    query = query.eq('age_group', options.ageGroup);
  }

  query = query
    .order('day_number', { ascending: true })
    .order('audio_duration', { ascending: true }) // Process shorter first
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
║           🏭 KELLY FACTORY - VIDEO GENERATION WORKER                     ║
╚══════════════════════════════════════════════════════════════════════════╝
`);

  // Parse CLI args
  const args = process.argv.slice(2);
  const options: {
    day?: number;
    limit?: number;
    ageGroup?: number;
    backend?: Backend;
  } = {};

  for (const arg of args) {
    if (arg.startsWith('--day=')) options.day = parseInt(arg.split('=')[1]);
    if (arg.startsWith('--limit=')) options.limit = parseInt(arg.split('=')[1]);
    if (arg.startsWith('--age=')) options.ageGroup = parseInt(arg.split('=')[1]);
    if (arg.startsWith('--backend=')) options.backend = arg.split('=')[1] as Backend;
  }

  // Check backends
  console.log('Available backends:');
  console.log(`  ${CONFIG.falKey ? '✅' : '❌'} Fal.ai (FAL_KEY)`);
  console.log(`  ${CONFIG.heygenKey ? '✅' : '❌'} HeyGen (HEYGEN_API_KEY)`);
  console.log(`  ⏸️  Local SadTalker (use batch script)\n`);

  if (!CONFIG.falKey && !CONFIG.heygenKey) {
    console.error('❌ No cloud backends available. Set FAL_KEY or HEYGEN_API_KEY.');
    process.exit(1);
  }

  // Get assets to process
  const assets = await getAssetsToProcess(options);
  
  if (assets.length === 0) {
    console.log('✨ No audio_ready assets to process.');
    
    // Show stats
    const { data: stats } = await supabase.rpc('get_factory_stats');
    if (stats?.[0]) {
      console.log(`\nFactory Status:`);
      console.log(`  Total: ${stats[0].total_assets}`);
      console.log(`  Complete: ${stats[0].complete}`);
      console.log(`  Audio Ready: ${stats[0].audio_ready}`);
      console.log(`  Pending: ${stats[0].pending}`);
    }
    return;
  }

  console.log(`Found ${assets.length} assets to process\n`);

  // Process assets
  let success = 0;
  let failed = 0;
  const startTime = Date.now();

  for (const asset of assets) {
    const result = await processAsset(asset, options.backend);
    if (result) success++;
    else failed++;
  }

  // Summary
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`
╔══════════════════════════════════════════════════════════════════════════╗
║                              📊 SUMMARY                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Processed: ${assets.length} assets
  ✅ Success: ${success}
  ❌ Failed: ${failed}
  ⏱️  Time: ${totalTime}s

${success > 0 ? '🎉 Videos generated and registered in kelly_lesson_assets!' : ''}
`);

  // Get updated stats
  const { data: finalStats } = await supabase.rpc('get_factory_stats');
  if (finalStats?.[0]) {
    console.log(`Factory Progress: ${finalStats[0].complete}/${finalStats[0].total_assets} (${finalStats[0].progress_pct}%)`);
  }
}

main().catch(console.error);
