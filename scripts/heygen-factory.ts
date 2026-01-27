#!/usr/bin/env npx tsx
/**
 * 🏭 HEYGEN FACTORY - Generate videos from audio_ready assets
 * 
 * Uses 660 HeyGen credits to create premium lip-synced Kelly videos.
 * Maps age_group to appropriate Kelly talking photo avatars.
 * 
 * Usage:
 *   npx tsx scripts/heygen-factory.ts              # Process up to 20 audio_ready assets
 *   npx tsx scripts/heygen-factory.ts --limit 50   # Process up to 50 assets
 *   npx tsx scripts/heygen-factory.ts --dry-run    # Preview without generating
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'heygen-factory'),
  BUCKET: 'kelly-videos',
  MAX_CREDITS: 660,
  POLL_INTERVAL_MS: 10000, // 10 seconds
  MAX_POLL_ATTEMPTS: 60,   // 10 minutes max wait
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// KELLY AVATAR MAPPING BY AGE GROUP
// Age groups: 8 (kid), 16 (teen), 35 (adult), 50 (mature), 70 (elder)
// =============================================================================

// Adult Kelly avatars (default for all ages until kid/teen are uploaded)
const ADULT_AVATARS: Record<string, string> = {
  architect: "afc54d3abfc04947bec026b9ec917ce8",
  diplomat: "433ad96bf5d647d9964cecf784d008f6",
  empath: "aa8b5eb1d711468a9a6e2085a4f8469c",
  explorer: "45e5ef8b651846e0b62b7477e552e87b",
  macgyver: "b9032c922c6e4e35b58a98abd499d060",
  mystic: "a2b31ed0b5f84b0fa02d15d411735d3a",
  provider: "06b78109ad22489ea2165ebbf180f77b",
  rebel: "e614671b193c40f99772f7de5d1c51f7",
  scientist: "7bb18cddacd44333813cc90ffa44f766",
  storyteller: "9ffd06bd986a4e3086612921f3ac87ea",
  strategist: "2411df8bdb0d40b088aa453d4c2a2d20",
  survivor: "3f44bd33bfd1494d916d2746808a1a39",
};

const ELDER_AVATARS: Record<string, string> = {
  scientist: "d2a5133b931541e986912a37139a9398",
  explorer: "5af13b2e9db14211a227f7e244b68e87",
  rebel: "62e4ea7a26524e60b04b35a190dbc023",
  architect: "b07df83db1bd4baaa7420ae792a6d35f",
  diplomat: "0abbbc925b144ade83a41d650d23ee10",
  empath: "6ed09093347d41f38f4d6638abd0a2c4",
  macgyver: "f24e88a269a54c17a2dffc19eec13123",
  mystic: "c76d77ffecc3461b87ea2fa0e21d719f",
  provider: "380af536b170462a907f7692a74367cc",
  storyteller: "817df044fe1c4f84a0de3aa00a296993",
  strategist: "e12c985879b94ef3955ee1fc95f30810",
  survivor: "a027f555728848a088324324c8f189e3",
};

const MATURE_AVATARS: Record<string, string> = {
  scientist: "6edf9b918f674e9dac2faa59d91f547c",
  explorer: "a762125d3107477aba43d1bd79f90d6e",
  rebel: "be06b44628864cb5acb86b81facb6323",
  architect: "7d3826b1d6e7451283f766b985fa65cf",
  diplomat: "d1d731dcdd5d4bb9af1c020a907671dc",
  empath: "871d1ff798214870961d1674bf87009f",
  macgyver: "687aa3d7ef1c4b55a955852ededb1a79",
  mystic: "d4eccf6a8d4c427b9313208d640db407",
  provider: "644411a0b2314928aed14d16ad4dd097",
  storyteller: "0727920ddd0a456090d009ff12258f3e",
  strategist: "9d66005de1ee436b92ffca5ec58fe213",
  survivor: "4d5b1893d5b84c9b8f17b90b1d4d9140",
};

// Default avatar to use (scientist - neutral/universal appeal)
const DEFAULT_AVATAR = ADULT_AVATARS.scientist;

function getAvatarForAge(ageGroup: number): string {
  // Map age_group to avatar set
  // Using 'scientist' archetype as default for all ages (most neutral)
  if (ageGroup >= 70) {
    return ELDER_AVATARS.scientist || DEFAULT_AVATAR;
  } else if (ageGroup >= 50) {
    return MATURE_AVATARS.scientist || DEFAULT_AVATAR;
  } else if (ageGroup >= 30) {
    return ADULT_AVATARS.scientist || DEFAULT_AVATAR;
  } else {
    // Kid (8) and Teen (16) - use adult for now until kid/teen avatars uploaded
    return ADULT_AVATARS.scientist || DEFAULT_AVATAR;
  }
}

// =============================================================================
// HEYGEN API FUNCTIONS
// =============================================================================

interface HeyGenVideoResponse {
  code: number;
  data: {
    video_id: string;
  };
  message: string | null;
}

interface HeyGenStatusResponse {
  code: number;
  data: {
    video_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    video_url?: string;
    thumbnail_url?: string;
    duration?: number;
    error?: string;
  };
}

async function submitHeyGenVideo(avatarId: string, audioUrl: string): Promise<string> {
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': CONFIG.HEYGEN_API_KEY,
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
      dimension: { width: 1080, height: 1080 }, // Square format for social/app
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HeyGen API error: ${response.status} - ${error}`);
  }

  const result: HeyGenVideoResponse = await response.json();
  
  if (result.code !== 100) {
    throw new Error(`HeyGen error: ${result.message || 'Unknown error'}`);
  }

  return result.data.video_id;
}

async function pollVideoStatus(videoId: string): Promise<{ url: string; duration: number }> {
  let attempts = 0;

  while (attempts < CONFIG.MAX_POLL_ATTEMPTS) {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      { headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY } }
    );

    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }

    const result: HeyGenStatusResponse = await response.json();

    if (result.data.status === 'completed') {
      return {
        url: result.data.video_url!,
        duration: result.data.duration || 0,
      };
    }

    if (result.data.status === 'failed') {
      throw new Error(`Video generation failed: ${result.data.error || 'Unknown error'}`);
    }

    process.stdout.write('.');
    await new Promise(r => setTimeout(r, CONFIG.POLL_INTERVAL_MS));
    attempts++;
  }

  throw new Error(`Timeout waiting for video ${videoId}`);
}

async function downloadAndUpload(
  videoUrl: string,
  dayNumber: number,
  phase: string,
  ageGroup: number
): Promise<string> {
  const response = await fetch(videoUrl);
  const buffer = Buffer.from(await response.arrayBuffer());

  const fileName = `day_${String(dayNumber).padStart(3, '0')}_${phase}_age${ageGroup}.mp4`;
  
  // Save locally for backup
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, fileName), buffer);

  // Upload to Supabase Storage
  const remotePath = `factory/day-${dayNumber}/${fileName}`;
  const { error } = await supabase.storage.from(CONFIG.BUCKET).upload(remotePath, buffer, {
    contentType: 'video/mp4',
    upsert: true,
  });

  if (error) {
    throw new Error(`Supabase upload failed: ${error.message}`);
  }

  const { data } = supabase.storage.from(CONFIG.BUCKET).getPublicUrl(remotePath);
  return data.publicUrl;
}

// =============================================================================
// MAIN FACTORY PIPELINE
// =============================================================================

interface AssetRecord {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  language: string;
  audio_url: string;
  audio_duration: number;
}

interface ProcessingResult {
  id: string;
  day_number: number;
  phase: string;
  age_group: number;
  status: 'success' | 'failed' | 'skipped';
  video_url?: string;
  error?: string;
  credits_used: number;
}

async function processAsset(asset: AssetRecord): Promise<ProcessingResult> {
  const { id, day_number, phase, age_group, audio_url } = asset;
  const result: ProcessingResult = {
    id,
    day_number,
    phase,
    age_group,
    status: 'failed',
    credits_used: 0,
  };

  try {
    // Get appropriate avatar
    const avatarId = getAvatarForAge(age_group);
    console.log(`\n🎬 Day ${day_number} | ${phase} | Age ${age_group}`);
    console.log(`   Avatar: ${avatarId}`);
    console.log(`   Audio: ${audio_url.substring(0, 60)}...`);

    // Submit to HeyGen
    console.log('   📤 Submitting to HeyGen...');
    const videoId = await submitHeyGenVideo(avatarId, audio_url);
    console.log(`   📹 Video ID: ${videoId}`);

    // Poll for completion
    process.stdout.write('   ⏳ Processing');
    const { url: heygenVideoUrl, duration } = await pollVideoStatus(videoId);
    console.log(' ✅');

    // Download and upload to Supabase
    console.log('   ☁️ Uploading to storage...');
    const videoUrl = await downloadAndUpload(heygenVideoUrl, day_number, phase, age_group);

    // Update database
    const { error } = await supabase
      .from('kelly_lesson_assets')
      .update({
        video_url: videoUrl,
        video_source: 'heygen',
        video_duration: duration,
        status: 'complete',
        updated_at: new Date().toISOString(),
      })
      .eq('id', id);

    if (error) {
      throw new Error(`Database update failed: ${error.message}`);
    }

    result.status = 'success';
    result.video_url = videoUrl;
    result.credits_used = 1; // Each video = 1 credit (estimate based on HeyGen pricing)
    console.log(`   ✅ Complete: ${videoUrl}`);

  } catch (error: any) {
    result.status = 'failed';
    result.error = error.message;
    console.error(`   ❌ Failed: ${error.message}`);

    // Update database with error
    await supabase
      .from('kelly_lesson_assets')
      .update({
        status: 'error',
        error_message: error.message,
        updated_at: new Date().toISOString(),
      })
      .eq('id', id);
  }

  return result;
}

async function verifyApiKey(): Promise<boolean> {
  try {
    // Test API key by listing talking photos
    const response = await fetch('https://api.heygen.com/v2/talking_photo.list', {
      headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY }
    });
    
    const data = await response.json();
    
    if (data.code === 400112 || data.message === 'Unauthorized') {
      return false;
    }
    
    return response.ok || data.code === 100;
  } catch {
    return false;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const isDryRun = args.includes('--dry-run');
  const limitArg = args.find(a => a.startsWith('--limit'));
  const limit = limitArg ? parseInt(args[args.indexOf(limitArg) + 1]) || 20 : 20;

  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🏭 HEYGEN FACTORY - Kelly Video Generation                ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log(`\n📊 Configuration:`);
  console.log(`   • Max credits available: ${CONFIG.MAX_CREDITS}`);
  console.log(`   • Processing limit: ${limit}`);
  console.log(`   • Mode: ${isDryRun ? '🔍 DRY RUN' : '🚀 PRODUCTION'}`);

  // Verify API key first
  if (!CONFIG.HEYGEN_API_KEY) {
    console.error('\n❌ HEYGEN_API_KEY not found in environment!');
    console.log('   Add it to .env: HEYGEN_API_KEY=your_key_here');
    console.log('   Get your key from: https://app.heygen.com/settings?nav=API');
    process.exit(1);
  }

  console.log('\n🔑 Verifying HeyGen API key...');
  const isValid = await verifyApiKey();
  
  if (!isValid) {
    console.error('\n❌ HeyGen API key is invalid or expired!');
    console.log('\n📋 To get a new API key:');
    console.log('   1. Go to https://app.heygen.com/settings?nav=API');
    console.log('   2. Click "Create API Key" or regenerate existing');
    console.log('   3. Copy the new key');
    console.log('   4. Update .env: HEYGEN_API_KEY=your_new_key');
    console.log('   5. Re-run this script');
    process.exit(1);
  }
  
  console.log('   ✅ API key valid');

  // Fetch audio_ready assets
  const { data: assets, error } = await supabase
    .from('kelly_lesson_assets')
    .select('id, day_number, phase, age_group, language, audio_url, audio_duration')
    .eq('status', 'audio_ready')
    .order('day_number', { ascending: true })
    .order('age_group', { ascending: true })
    .limit(limit);

  if (error) {
    console.error(`\n❌ Database error: ${error.message}`);
    process.exit(1);
  }

  if (!assets || assets.length === 0) {
    console.log('\n⚠️ No audio_ready assets found!');
    console.log('   Run the audio generation pipeline first.');
    process.exit(0);
  }

  console.log(`\n📋 Found ${assets.length} audio_ready assets:`);
  assets.forEach((a: AssetRecord) => {
    console.log(`   • Day ${a.day_number} | ${a.phase} | Age ${a.age_group} | ${a.audio_duration}s`);
  });

  if (isDryRun) {
    console.log('\n🔍 Dry run complete. No videos generated.');
    process.exit(0);
  }

  // Process assets
  console.log('\n' + '═'.repeat(60));
  console.log('🚀 STARTING VIDEO GENERATION');
  console.log('═'.repeat(60));

  const results: ProcessingResult[] = [];

  for (const asset of assets as AssetRecord[]) {
    const result = await processAsset(asset);
    results.push(result);

    // Brief delay between submissions to avoid rate limiting
    await new Promise(r => setTimeout(r, 2000));
  }

  // Generate report
  console.log('\n\n' + '═'.repeat(60));
  console.log('📋 FINAL REPORT');
  console.log('═'.repeat(60));

  const success = results.filter(r => r.status === 'success');
  const failed = results.filter(r => r.status === 'failed');
  const totalCredits = success.reduce((sum, r) => sum + r.credits_used, 0);

  console.log(`\n✅ Videos generated: ${success.length}`);
  console.log(`❌ Failed: ${failed.length}`);
  console.log(`💳 HeyGen credits used: ${totalCredits}/${CONFIG.MAX_CREDITS}`);
  console.log(`📊 Remaining credits: ${CONFIG.MAX_CREDITS - totalCredits}`);

  if (failed.length > 0) {
    console.log('\n❌ Failed assets:');
    failed.forEach(f => {
      console.log(`   • Day ${f.day_number} | ${f.phase} | Age ${f.age_group}: ${f.error}`);
    });
  }

  if (success.length > 0) {
    console.log('\n🎬 Sample video URLs for quality check:');
    success.slice(0, 3).forEach(s => {
      console.log(`   • ${s.video_url}`);
    });
  }

  // Save results to file
  const reportPath = path.join(CONFIG.OUTPUT_DIR, `factory_report_${Date.now()}.json`);
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify({ results, summary: {
    total: results.length,
    success: success.length,
    failed: failed.length,
    credits_used: totalCredits,
    timestamp: new Date().toISOString(),
  }}, null, 2));
  console.log(`\n📄 Report saved: ${reportPath}`);

  console.log('\n🎯 Factory run complete!');
}

main().catch(console.error);
