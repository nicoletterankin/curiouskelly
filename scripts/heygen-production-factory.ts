#!/usr/bin/env npx tsx
/**
 * 🏭 HEYGEN PRODUCTION FACTORY
 * 
 * Generates lip-synced Kelly videos using avatar_id + look_id format.
 * 
 * Usage:
 *   npx tsx scripts/heygen-production-factory.ts --limit=1   # Test with 1
 *   npx tsx scripts/heygen-production-factory.ts --limit=50  # Production batch
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// =============================================================================
// KELLY AVATAR IDS - VERIFIED FROM HEYGEN DASHBOARD
// See docs/KELLY_AVATAR_IDS.md for full reference
// =============================================================================

// Avatar Group IDs (by age)
const AVATAR_GROUPS = {
  kid: 'a762125d3107477aba43d1bd79f90d6e',    // ages <= 12
  adult: 'a762125d3107477aba43d1bd79f90d6e',   // ages 13-54
  senior: 'd8c4ffac39a546a682b603c56e15906a',  // ages 55+
} as const;

// Look IDs by age group and archetype
const ADULT_LOOKS: Record<string, string> = {
  storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
  explorer: '62516885ca4b4eae8f63b87b8c060e25',
  scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
  architect: '35d0115505824e3182eb9d2ee8cfe73d',
  strategist: '08d53d1b065041bda2e5b6bc32962a8a',
  diplomat: 'c3cdbe48fe274420a7f45a4da7e366aa',
  mystic: 'dfaf9fbd644a475595b178f0be65a39a',
  rebel: '390be3fb2b064883bb2304fc3968fd87',
  macgyver: 'c5aab6ab13d940f8ae4700d546bd6b6b',
  empath: '6bb1a05678c64213a1ed3a4dc790b81e',
  provider: '9a143feeb2994989b034cebeb78753be',
  survivor: '831c8d6048104ba0b03a74a36543cfb9',
};

const KID_LOOKS: Record<string, string> = {
  scientist: '82813816115c4fbe93b3f3f211bd9931',
  explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
  rebel: 'd4e960f7a3424d869877f3a951adfae7',
  architect: 'cc1dd0e9e2fd432099985c9b036ed836',
  diplomat: '48bddc41ae94473caa645ce9ab93136d',
  empath: 'deeb27f2648848b48c5c1ce59059bd54',
  macgyver: '7b6ab196f2c7430b945411df51a84c58',
  mystic: '5cff601bfb344015a65ff46c6b8cd70a',
  provider: 'deaa213342944dc2bf671abe1442e316',
  storyteller: '1024bc304a1146998bc4c360173b2c48',
  strategist: '6249632f58ce479891de00b4da5fb88d',
  survivor: 'bd579e4ca77444aca2bfea8ee9070830',
};

const SENIOR_LOOKS: Record<string, string> = {
  scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
  explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
  architect: '42e9197ab9d84961915b00d5cc780190',
  empath: '493dac2cf2ba4509b3cc048ff819765e',
  diplomat: 'a82183881e284e3782db75b755c3f080',
  macgyver: 'cb5b025506284d64b696e296ca2feead',
  provider: '12582467e9ff48889d7b2435642e2d65',
  storyteller: '98178c87897e4421884b535b7864ba86',
  strategist: 'e4ab0d4d1f1b4dc9b81a1076b018557f',
  rebel: 'dc835263eaa247f5b0e06106b848df18',
  mystic: 'c6d104b2ca354b0a9593cb840988bf6e',
  survivor: '9a143feeb2994989b034cebeb78753be',
};

// Get avatar_id and look_id for an age group
function getKellyAvatar(ageGroup: number, archetype: string = 'storyteller'): { avatar_id: string; look_id: string } {
  let avatarId: string;
  let lookMap: Record<string, string>;
  
  if (ageGroup <= 12) {
    avatarId = '93bb788b97d847409ad7dcf69702ece5'; // kid
    lookMap = KID_LOOKS;
  } else if (ageGroup >= 55) {
    avatarId = 'd8c4ffac39a546a682b603c56e15906a'; // senior
    lookMap = SENIOR_LOOKS;
  } else {
    avatarId = 'a762125d3107477aba43d1bd79f90d6e'; // adult
    lookMap = ADULT_LOOKS;
  }
  
  return {
    avatar_id: avatarId,
    look_id: lookMap[archetype] || lookMap['storyteller']
  };
}

interface LessonAsset {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  language: string;
  audio_url: string;
}

// =============================================================================
// STEP 1: Create video with HeyGen (avatar_id + look_id format)
// =============================================================================

async function createVideo(asset: LessonAsset): Promise<string> {
  console.log(`[HeyGen] Creating video for Day ${asset.day_number}/${asset.phase} age${asset.age_group}...`);
  
  const { avatar_id, look_id } = getKellyAvatar(asset.age_group, 'storyteller');
  console.log(`[HeyGen] Using avatar_id: ${avatar_id}, look_id: ${look_id}`);

  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'avatar',
          avatar_id: avatar_id,
          look_id: look_id,
          avatar_style: 'normal'
        },
        voice: {
          type: 'audio',
          audio_url: asset.audio_url
        },
        background: {
          type: 'color',
          value: '#FFFFFF'
        }
      }],
      dimension: { width: 1080, height: 1920 },
      aspect_ratio: '9:16',
      test: false  // Use real credits, not test mode
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HeyGen API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  
  if (data.error) {
    throw new Error(`HeyGen error: ${data.error.message}`);
  }
  
  console.log(`[HeyGen] Video queued: ${data.data.video_id}`);
  return data.data.video_id;
}

// =============================================================================
// STEP 2: Poll for completion
// =============================================================================

async function waitForVideo(videoId: string, maxWaitMs = 600000): Promise<string> {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
    );

    const data = await response.json();
    const status = data.data?.status;

    console.log(`[HeyGen] Status: ${status}`);

    if (status === 'completed') {
      return data.data.video_url;
    }

    if (status === 'failed') {
      throw new Error(`Video generation failed: ${data.data.error}`);
    }

    // Wait 10 seconds before next poll
    await new Promise(r => setTimeout(r, 10000));
  }

  throw new Error('Video generation timed out');
}

// =============================================================================
// STEP 3: Update Supabase
// =============================================================================

async function updateRegistry(assetId: string, videoUrl: string): Promise<void> {
  const { error } = await supabase
    .from('kelly_lesson_assets')
    .update({
      video_url: videoUrl,
      video_source: 'heygen',
      status: 'complete',
      updated_at: new Date().toISOString()
    })
    .eq('id', assetId);

  if (error) throw error;
  console.log(`[Registry] Updated: ${assetId} → complete`);
}

// =============================================================================
// MAIN FACTORY LOOP
// =============================================================================

async function runFactory(limit: number = 10) {
  console.log('========================================');
  console.log('🏭 HEYGEN PRODUCTION FACTORY');
  console.log('   Using avatar_id + look_id format');
  console.log('========================================');

  // Validate environment
  if (!HEYGEN_API_KEY) {
    console.error('❌ HEYGEN_API_KEY not set in environment');
    process.exit(1);
  }

  // Test API key and show quota
  console.log('\n🔑 Testing HeyGen API key...');
  const quotaRes = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const quotaData = await quotaRes.json();
  
  if (quotaData.error) {
    console.error(`❌ API key invalid: ${quotaData.error.message}`);
    process.exit(1);
  }
  
  console.log(`✅ API key valid. Credits: ${quotaData.data?.remaining_quota || 'unknown'}`);

  console.log(`\n📊 Configuration:`);
  console.log(`   • Limit: ${limit} videos`);
  console.log(`   • Using avatar_id + look_id format`);
  console.log(`   • Default archetype: storyteller`);

  // Get audio_ready assets
  const { data: assets, error } = await supabase
    .from('kelly_lesson_assets')
    .select('*')
    .eq('status', 'audio_ready')
    .not('audio_url', 'is', null)
    .order('day_number', { ascending: true })
    .limit(limit);

  if (error) throw error;

  console.log(`\n📋 Found ${assets?.length || 0} assets to process`);

  if (!assets || assets.length === 0) {
    console.log('⚠️ No assets ready for video generation');
    return;
  }

  // List assets
  assets.forEach((a: LessonAsset) => {
    const { avatar_id } = getKellyAvatar(a.age_group);
    const ageLabel = a.age_group <= 12 ? 'kid' : a.age_group >= 55 ? 'senior' : 'adult';
    console.log(`   • Day ${a.day_number} | ${a.phase} | ${ageLabel} (${a.age_group})`);
  });

  let success = 0;
  let failed = 0;

  for (const asset of assets as LessonAsset[]) {
    try {
      console.log(`\n--- Processing Day ${asset.day_number}/${asset.phase} ---`);

      // Create video
      const videoId = await createVideo(asset);

      // Wait for completion
      const heygenUrl = await waitForVideo(videoId);

      // Update registry
      await updateRegistry(asset.id, heygenUrl);

      console.log(`✅ COMPLETE: ${heygenUrl}`);
      success++;

      // Rate limit: wait 2 seconds between videos
      await new Promise(r => setTimeout(r, 2000));

    } catch (err: any) {
      console.error(`❌ FAILED Day ${asset.day_number}/${asset.phase}:`, err.message);
      failed++;

      // Mark as error in Supabase
      await supabase
        .from('kelly_lesson_assets')
        .update({ 
          status: 'error', 
          error_message: String(err.message || err),
          updated_at: new Date().toISOString()
        })
        .eq('id', asset.id);
    }
  }

  console.log('\n========================================');
  console.log(`🏭 FACTORY COMPLETE: ${success} success, ${failed} failed`);
  console.log('========================================');
}

// =============================================================================
// CLI
// =============================================================================

const args = process.argv.slice(2);
const limitArg = args.find(a => a.startsWith('--limit='));
const limit = limitArg ? parseInt(limitArg.split('=')[1]) : 10;

runFactory(limit).catch(console.error);
