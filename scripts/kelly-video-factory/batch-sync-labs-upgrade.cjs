#!/usr/bin/env node
/**
 * Batch upgrade existing videos to Sync Labs quality (95% accuracy)
 * 
 * Usage:
 *   node batch-sync-labs-upgrade.cjs --days 1-5
 *   node batch-sync-labs-upgrade.cjs --day 1 --phase hook
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const SYNC_LABS_API_KEY = process.env.SYNC_LABS_API_KEY;
const SYNC_LABS_API_URL = 'https://api.sync.so/v2';

// Rate limiting
const CONCURRENT_JOBS = 3;
const DELAY_BETWEEN_JOBS = 2000; // 2 seconds

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function submitSyncLabsJob(videoUrl, audioUrl) {
  const response = await fetch(`${SYNC_LABS_API_URL}/generate`, {
    method: 'POST',
    headers: {
      'x-api-key': SYNC_LABS_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [
        { type: 'video', url: videoUrl },
        { type: 'audio', url: audioUrl }
      ]
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Sync Labs API error: ${response.status} - ${error}`);
  }

  return response.json();
}

async function pollSyncLabsJob(jobId, maxAttempts = 120) {
  for (let i = 0; i < maxAttempts; i++) {
    const response = await fetch(`${SYNC_LABS_API_URL}/generate/${jobId}`, {
      headers: { 'x-api-key': SYNC_LABS_API_KEY }
    });

    const job = await response.json();

    if (job.status === 'COMPLETED') {
      return job.output?.[0]?.url || job.outputUrl || job.output;
    }

    if (job.status === 'FAILED' || job.status === 'REJECTED') {
      throw new Error(`Sync Labs job failed: ${job.error || job.message}`);
    }

    if (i % 12 === 0) {
      console.log(`      Status: ${job.status} (${Math.round(i * 5 / 60)}m)`);
    }
    process.stdout.write('.');
    await sleep(5000);
  }

  throw new Error('Job timed out after 10 minutes');
}

async function getAssetsToUpgrade(dayStart, dayEnd, phaseFilter) {
  // Get unique day/phase combinations with their animation and audio
  let query = supabase
    .from('kelly_video_assets')
    .select('day_number, phase, public_url, asset_type')
    .in('asset_type', ['animation', 'audio'])
    .gte('day_number', dayStart)
    .lte('day_number', dayEnd)
    .order('day_number')
    .order('phase');

  if (phaseFilter) {
    query = query.eq('phase', phaseFilter);
  }

  const { data, error } = await query;
  if (error) throw error;

  // Group by day/phase
  const grouped = {};
  for (const row of data) {
    const key = `${row.day_number}_${row.phase}`;
    if (!grouped[key]) {
      grouped[key] = { day_number: row.day_number, phase: row.phase };
    }
    if (row.asset_type === 'animation') {
      grouped[key].animation_url = row.public_url;
    } else if (row.asset_type === 'audio') {
      // Get first audio (we'll pick one archetype)
      if (!grouped[key].audio_url) {
        grouped[key].audio_url = row.public_url;
      }
    }
  }

  // Filter to only those with both animation and audio
  return Object.values(grouped).filter(g => g.animation_url && g.audio_url);
}

async function upgradeVideo(asset) {
  const { day_number, phase, animation_url, audio_url } = asset;
  
  console.log(`\n🎬 Upgrading Day ${day_number} ${phase}`);
  console.log(`   Animation: ${animation_url.substring(0, 60)}...`);
  console.log(`   Audio: ${audio_url.substring(0, 60)}...`);

  try {
    // Submit to Sync Labs
    console.log('   🚀 Submitting to Sync Labs...');
    const job = await submitSyncLabsJob(animation_url, audio_url);
    console.log(`   Job ID: ${job.id}`);

    // Poll for completion
    console.log('   ⏳ Processing...');
    const resultUrl = await pollSyncLabsJob(job.id);
    console.log(`\n   ✅ Complete!`);
    console.log(`   Output: ${resultUrl}`);

    // Save to database
    const storagePath = `production/videos-hq/day_${String(day_number).padStart(3, '0')}_${phase}_sync.mp4`;
    
    await supabase.from('kelly_video_assets').insert({
      day_number,
      phase,
      template: phase,
      asset_type: 'video_hq',
      quality_tier: 'production',
      storage_path: storagePath,
      public_url: resultUrl,
      generation_cost_usd: 0.25, // Estimated Sync Labs cost
      status: 'generated'
    });

    return { success: true, day_number, phase, url: resultUrl };

  } catch (error) {
    console.log(`   ❌ Error: ${error.message}`);
    return { success: false, day_number, phase, error: error.message };
  }
}

async function main() {
  const args = process.argv.slice(2);
  
  // Parse arguments
  let dayStart = 1, dayEnd = 5;
  let phaseFilter = null;
  
  const daysIndex = args.indexOf('--days');
  if (daysIndex > -1) {
    const [start, end] = args[daysIndex + 1].split('-').map(Number);
    dayStart = start;
    dayEnd = end || start;
  }
  
  const dayIndex = args.indexOf('--day');
  if (dayIndex > -1) {
    dayStart = dayEnd = parseInt(args[dayIndex + 1]);
  }

  const phaseIndex = args.indexOf('--phase');
  if (phaseIndex > -1) {
    phaseFilter = args[phaseIndex + 1];
  }

  console.log('═'.repeat(70));
  console.log('🚀 SYNC LABS BATCH UPGRADE');
  console.log(`   Days ${dayStart}-${dayEnd} → 95% accuracy lip-sync`);
  if (phaseFilter) console.log(`   Phase filter: ${phaseFilter}`);
  console.log('═'.repeat(70));

  if (!SYNC_LABS_API_KEY) {
    console.error('❌ SYNC_LABS_API_KEY not configured');
    process.exit(1);
  }

  // Get assets to upgrade
  const assets = await getAssetsToUpgrade(dayStart, dayEnd, phaseFilter);
  console.log(`\n📊 Found ${assets.length} day/phase combinations to upgrade`);

  if (assets.length === 0) {
    console.log('Nothing to upgrade. Make sure animations and audio exist.');
    return;
  }

  // Process in batches
  const results = [];
  for (let i = 0; i < assets.length; i++) {
    const result = await upgradeVideo(assets[i]);
    results.push(result);
    
    // Rate limit
    if (i < assets.length - 1) {
      await sleep(DELAY_BETWEEN_JOBS);
    }
  }

  // Summary
  console.log('\n');
  console.log('═'.repeat(70));
  console.log('📊 UPGRADE SUMMARY');
  console.log('═'.repeat(70));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`   ✅ Successful: ${successful.length}`);
  console.log(`   ❌ Failed: ${failed.length}`);
  
  if (failed.length > 0) {
    console.log('\n   Failed items:');
    failed.forEach(f => {
      console.log(`      Day ${f.day_number} ${f.phase}: ${f.error}`);
    });
  }

  console.log('═'.repeat(70));
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});



